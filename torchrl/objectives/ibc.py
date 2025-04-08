from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

from torchrl.modules.planners.stochastic_optimizers import AutoRegressiveDFO, DFO, MCMC
from torchrl.objectives.common import LossModule


class IBCLoss(LossModule):
    """Implicit Behavioral Cloning (IBC) loss.

    This loss function implements the IBC algorithm as described in the paper
    "Implicit Behavioral Cloning" (https://arxiv.org/abs/2109.00137).

    Args:
        ebm_network (TensorDictModule): The energy-based model network that maps observations and actions to energy values.
        optimizer (Union[MCMC, AutoRegressiveDFO, DFO]): The stochastic optimizer used for sampling.
        observation_key (str, optional): The key for observations in the tensordict. Defaults to "observation".
        action_key (str, optional): The key for actions in the tensordict. Defaults to "action".
        energy_key (str, optional): The key for energy values in the tensordict. Defaults to "energy".

    Examples:
        >>> from torchrl.modules import TensorDictModule
        >>> from torchrl.modules.planners.stochastic_optimizers import DFO
        >>> # Create a simple EBM network
        >>> class EBM(nn.Module):
        ...     def __init__(self, obs_dim, action_dim):
        ...         super().__init__()
        ...         self.net = nn.Sequential(
        ...             nn.Linear(obs_dim + action_dim, 64),
        ...             nn.ReLU(),
        ...             nn.Linear(64, 1)
        ...         )
        ...     def forward(self, obs, action):
        ...         x = torch.cat([obs, action], dim=-1)
        ...         return self.net(x).squeeze(-1)
        >>> # Create the EBM network and optimizer
        >>> ebm = TensorDictModule(EBM(4, 2), in_keys=["observation", "action"], out_keys=["energy"])
        >>> optimizer = DFO()
        >>> # Create the loss
        >>> loss = IBCLoss(ebm, optimizer)
        >>> # Use the loss
        >>> td = TensorDict({"observation": torch.randn(32, 4), "action": torch.randn(32, 2)}, [32])
        >>> loss(td)
        TensorDict(
            fields={
                loss_ebm: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.float32, is_shared=False),
                loss: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            energy (NestedKey): The input tensordict key where the
                energy value is expected. Defaults to ``"energy"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Defaults to ``"terminated"``.
        """

        action: NestedKey = "action"
        priority: NestedKey = "priority"
        energy: NestedKey = "energy"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    out_keys = [
        "loss_ebm",
    ]

    ebm_network: TensorDictModule
    optimizer: MCMC | AutoRegressiveDFO | DFO
    observation_key: str
    action_key: str
    energy_key: str

    def __init__(
        self,
        ebm_network: TensorDictModule,
        optimizer: MCMC | AutoRegressiveDFO | DFO,
        observation_key: str = "observation",
        action_key: str = "action",
        energy_key: str = "energy",
    ):
        self._in_keys = None
        self._out_keys = None
        super().__init__()

        self.ebm_network = ebm_network
        self.optimizer = optimizer
        self.observation_key = observation_key
        self.action_key = action_key
        self.energy_key = energy_key

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.ebm_network.in_keys,
            *[("next", key) for key in self.ebm_network.in_keys],
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Forward pass of the loss function.

        Args:
            tensordict (TensorDictBase): The input tensordict containing observations and actions.

        Returns:
            TensorDictBase: The tensordict with the computed losses.
        """
        # Get observations and actions
        obs = tensordict.get(self.observation_key)
        action = tensordict.get(self.action_key)

        # Handle single dimension cases
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(-1)
        if len(action.shape) == 1:
            action = action.unsqueeze(-1)

        # Sample negative actions using the optimizer
        obs, negatives = self.optimizer.sample(obs, self.ebm_network)

        # Combine current and negative actions
        targets = torch.cat([action.unsqueeze(dim=1), negatives], dim=1)
        obs = torch.cat([obs[:, :1], obs], dim=1)

        # Random permutation for InfoNCE
        permutation = torch.rand(targets.shape[0], targets.shape[1]).argsort(dim=1)
        targets = targets[torch.arange(targets.shape[0]).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(obs.device)

        # Compute energy and logits
        energy = self.ebm_network(obs, targets)
        if energy.shape[-1] == 1 and len(energy.shape) > 1:
            energy = energy.squeeze(-1)
        logits = -1.0 * energy

        # Handle different optimizer types
        if isinstance(self.optimizer, AutoRegressiveDFO):
            # autoregressive case
            ground_truth = torch.unsqueeze(ground_truth, -1).repeat(
                1, 1, logits.shape[-1]
            )
        loss = F.cross_entropy(logits, ground_truth)
        loss_dict = {"loss_ebm": loss}

        if isinstance(self.optimizer, MCMC):
            # MCMC case with gradient penalty
            grad_penalty = self.optimizer.grad_penalty(obs, targets, self.ebm_network)
            loss += grad_penalty
            loss_dict["grad_penalty"] = grad_penalty

        # Set total loss and priority
        loss_dict["loss"] = loss

        tensordict.set(self.tensor_keys.priority, energy.mean(dim=-1))

        return TensorDict(loss_dict)
