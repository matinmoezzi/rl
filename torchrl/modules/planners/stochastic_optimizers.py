from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def unsqueeze_repeat(
    x: torch.Tensor, repeat_times: int, unsqueeze_dim: int = 0
) -> torch.Tensor:
    """Squeeze the tensor on `unsqueeze_dim` and then repeat in this dimension for `repeat_times` times.

    This is useful for preprocessing the input to a model ensemble.

    Args:
        x: The tensor to squeeze and repeat
        repeat_times: The number of times to repeat the tensor
        unsqueeze_dim: The dimension to unsqueeze

    Returns:
        The unsqueezed and repeated tensor

    Examples:
        >>> x = torch.ones(64, 6)
        >>> x = unsqueeze_repeat(x, 4)
        >>> x.shape == (4, 64, 6)

        >>> x = torch.ones(64, 6)
        >>> x = unsqueeze_repeat(x, 4, -1)
        >>> x.shape == (64, 6, 4)
    """
    if not -1 <= unsqueeze_dim <= len(x.shape):
        raise ValueError(f"unsqueeze_dim should be from {-1} to {len(x.shape)}")

    x = x.unsqueeze(unsqueeze_dim)
    repeats = [1] * len(x.shape)
    repeats[unsqueeze_dim] *= repeat_times
    return x.repeat(*repeats)


def no_ebm_grad():
    """Decorator that temporarily disables gradients for the energy-based model.

    This is used to prevent gradient computation through the EBM during certain operations
    while ensuring gradients are re-enabled afterward.
    """

    def ebm_disable_grad_wrapper(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ebm = args[-1]
            if not isinstance(ebm, nn.Module):
                raise TypeError(
                    "Make sure ebm is the last positional argument and is a nn.Module."
                )

            ebm.requires_grad_(False)
            result = func(*args, **kwargs)
            ebm.requires_grad_(True)
            return result

        return wrapper

    return ebm_disable_grad_wrapper


class StochasticOptimizer(ABC):
    """Base class for stochastic optimizers.

    This abstract class defines the interface for stochastic optimizers used in
    energy-based models for action sampling and inference.
    """

    def __init__(self, device: str = "cpu"):
        """Initialize the stochastic optimizer.

        Args:
            device: The device to use for tensor operations
        """
        self.action_bounds: Optional[torch.Tensor] = None
        self.device = device

    def _sample(
        self, obs: torch.Tensor, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw action samples from the uniform random distribution and tile observations.

        Args:
            obs: Observation tensor of shape (B, O)
            num_samples: The number of samples to generate

        Returns:
            A tuple containing:
                - tiled_obs: Observations tiled to shape (B, N, O)
                - action_samples: Action samples of shape (B, N, A)

        Raises:
            RuntimeError: If action_bounds is not set
        """
        if self.action_bounds is None:
            raise RuntimeError("Action bounds must be set before sampling")

        size = (obs.shape[0], num_samples, self.action_bounds.shape[1])
        low, high = self.action_bounds[0, :], self.action_bounds[1, :]
        action_samples = low + (high - low) * torch.rand(size, device=self.device)
        tiled_obs = unsqueeze_repeat(obs, num_samples, 1)
        return tiled_obs, action_samples

    @staticmethod
    @torch.no_grad()
    def _get_best_action_sample(
        obs: torch.Tensor, action_samples: torch.Tensor, ebm: nn.Module
    ) -> torch.Tensor:
        """Return one action for each batch with highest probability (lowest energy).

        Args:
            obs: Observation tensor of shape (B, O)
            action_samples: Action samples of shape (B, N, A)
            ebm: Energy-based model

        Returns:
            Best action samples of shape (B, A)
        """
        # (B, N)
        energies = ebm.forward(obs, action_samples)
        if energies.shape[-1] == 1 and len(energies.shape) > 1:
            energies = energies.squeeze(-1)
        probs = F.softmax(-1.0 * energies, dim=-1)
        # (B, )
        best_idxs = probs.argmax(dim=-1)
        return action_samples[
            torch.arange(action_samples.size(0), device=action_samples.device),
            best_idxs,
        ]

    def set_action_bounds(self, action_bounds: np.ndarray) -> None:
        """Set action bounds calculated from the dataset statistics.

        Args:
            action_bounds: Array of shape (2, A), where action_bounds[0] is lower bound
                          and action_bounds[1] is upper bound
        """
        self.action_bounds = torch.as_tensor(
            action_bounds, dtype=torch.float32, device=self.device
        )

    @abstractmethod
    def sample(
        self, obs: torch.Tensor, ebm: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create tiled observations and sample counter-negatives for InfoNCE loss.

        Args:
            obs: Observations of shape (B, O)
            ebm: Energy-based model

        Returns:
            A tuple containing:
                - tiled_obs: Tiled observations of shape (B, N, O)
                - action_samples: Action samples of shape (B, N, A)
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation.

        Args:
            obs: Observations of shape (B, O)
            ebm: Energy-based model

        Returns:
            Best action samples of shape (B, A)
        """
        raise NotImplementedError


class DFO(StochasticOptimizer):
    """Derivative-Free Optimizer as described in Implicit Behavioral Cloning.

    Reference: https://arxiv.org/abs/2109.00137
    """

    def __init__(
        self,
        noise_scale: float = 0.33,
        noise_shrink: float = 0.5,
        iters: int = 3,
        train_samples: int = 8,
        inference_samples: int = 16384,
        device: str = "cpu",
    ):
        """Initialize the Derivative-Free Optimizer.

        Args:
            noise_scale: Initial noise scale
            noise_shrink: Noise scale shrink rate
            iters: Number of iterations
            train_samples: Number of samples for training
            inference_samples: Number of samples for inference
            device: Device to use for tensor operations
        """
        super().__init__(device)
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.iters = iters
        self.train_samples = train_samples
        self.inference_samples = inference_samples

    def sample(
        self, obs: torch.Tensor, ebm: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from uniform distribution.

        Args:
            obs: Observations of shape (B, O)
            ebm: Energy-based model

        Returns:
            A tuple containing:
                - tiled_obs: Tiled observations of shape (B, N, O)
                - action_samples: Action samples of shape (B, N, A)
        """
        return self._sample(obs, self.train_samples)

    @torch.no_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action using derivative-free optimization.

        Args:
            obs: Observations of shape (B, O)
            ebm: Energy-based model

        Returns:
            Best action samples of shape (B, A)
        """
        noise_scale = self.noise_scale

        # (B, N, O), (B, N, A)
        obs, action_samples = self._sample(obs, self.inference_samples)

        for _ in range(self.iters):
            # (B, N)
            energies = ebm.forward(obs, action_samples)
            if energies.shape[-1] == 1 and len(energies.shape) > 1:
                energies = energies.squeeze(-1)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            action_samples = action_samples[
                torch.arange(
                    action_samples.size(0), device=action_samples.device
                ).unsqueeze(-1),
                idxs,
            ]

            # Add noise and clip to target bounds
            action_samples = (
                action_samples + torch.randn_like(action_samples) * noise_scale
            )
            action_samples = action_samples.clamp(
                min=self.action_bounds[0, :], max=self.action_bounds[1, :]
            )

            noise_scale *= self.noise_shrink

        # Return target with highest probability
        return self._get_best_action_sample(obs, action_samples, ebm)


class AutoRegressiveDFO(DFO):
    """AutoRegressive Derivative-Free Optimizer as described in Implicit Behavioral Cloning.

    Reference: https://arxiv.org/abs/2109.00137
    """

    def __init__(
        self,
        noise_scale: float = 0.33,
        noise_shrink: float = 0.5,
        iters: int = 3,
        train_samples: int = 8,
        inference_samples: int = 4096,
        device: str = "cpu",
    ):
        """Initialize the AutoRegressive Derivative-Free Optimizer.

        Args:
            noise_scale: Initial noise scale
            noise_shrink: Noise scale shrink rate
            iters: Number of iterations
            train_samples: Number of samples for training
            inference_samples: Number of samples for inference
            device: Device to use for tensor operations
        """
        super().__init__(
            noise_scale, noise_shrink, iters, train_samples, inference_samples, device
        )

    @torch.no_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action using autoregressive derivative-free optimization.

        Args:
            obs: Observations of shape (B, O)
            ebm: Energy-based model

        Returns:
            Best action samples of shape (B, A)
        """
        noise_scale = self.noise_scale

        # (B, N, O), (B, N, A)
        obs, action_samples = self._sample(obs, self.inference_samples)

        for _ in range(self.iters):
            # j: action_dim index
            for j in range(action_samples.shape[-1]):
                # (B, N)
                energies = ebm.forward(obs, action_samples)
                if energies.shape[-1] == 1 and len(energies.shape) > 1:
                    energies = energies.squeeze(-1)
                energies = energies[..., j]
                probs = F.softmax(-1.0 * energies, dim=-1)

                # Resample with replacement
                idxs = torch.multinomial(
                    probs, self.inference_samples, replacement=True
                )
                action_samples = action_samples[
                    torch.arange(
                        action_samples.size(0), device=action_samples.device
                    ).unsqueeze(-1),
                    idxs,
                ]

                # Add noise and clip to target bounds
                action_samples[..., j] = (
                    action_samples[..., j]
                    + torch.randn_like(action_samples[..., j]) * noise_scale
                )
                action_samples[..., j] = action_samples[..., j].clamp(
                    min=self.action_bounds[0, j], max=self.action_bounds[1, j]
                )

            noise_scale *= self.noise_shrink

        # (B, N)
        energies = ebm.forward(obs, action_samples)
        if energies.shape[-1] == 1 and len(energies.shape) > 1:
            energies = energies.squeeze(-1)
        probs = F.softmax(-1.0 * energies, dim=-1)
        # (B, )
        best_idxs = probs.argmax(dim=-1)
        return action_samples[
            torch.arange(action_samples.size(0), device=action_samples.device),
            best_idxs,
        ]


class MCMC(StochasticOptimizer):
    """MCMC method as stochastic optimizers in Implicit Behavioral Cloning.

    Reference: https://arxiv.org/abs/2109.00137
    """

    class BaseScheduler(ABC):
        """Base class for learning rate schedulers."""

        @abstractmethod
        def get_rate(self, index: int) -> float:
            """Get learning rate for the given index.

            Args:
                index: Current iteration index

            Returns:
                Learning rate value
            """
            raise NotImplementedError

    class ExponentialScheduler(BaseScheduler):
        """Exponential learning rate scheduler for Langevin sampler."""

        def __init__(self, init: float, decay: float):
            """Initialize the ExponentialScheduler.

            Args:
                init: Initial learning rate
                decay: Decay rate
            """
            self._decay = decay
            self._latest_lr = init

        def get_rate(self, index: int) -> float:
            """Get learning rate. Assumes calling sequentially.

            Args:
                index: Current iteration index (unused)

            Returns:
                Current learning rate
            """
            lr = self._latest_lr
            self._latest_lr *= self._decay
            return lr

    class PolynomialScheduler(BaseScheduler):
        """Polynomial learning rate scheduler for Langevin sampler."""

        def __init__(self, init: float, final: float, power: float, num_steps: int):
            """Initialize the PolynomialScheduler.

            Args:
                init: Initial learning rate
                final: Final learning rate
                power: Power of polynomial
                num_steps: Number of steps
            """
            self._init = init
            self._final = final
            self._power = power
            self._num_steps = num_steps

        def get_rate(self, index: int) -> float:
            """Get learning rate for the given index.

            Args:
                index: Current iteration index

            Returns:
                Current learning rate
            """
            if index == -1:
                return self._init
            return (
                (self._init - self._final)
                * ((1 - (float(index) / float(self._num_steps - 1))) ** (self._power))
            ) + self._final

    def __init__(
        self,
        iters: int = 100,
        use_langevin_negative_samples: bool = True,
        train_samples: int = 8,
        inference_samples: int = 512,
        stepsize_scheduler: Dict[str, float] = None,
        optimize_again: bool = True,
        again_stepsize_scheduler: Dict[str, float] = None,
        device: str = "cpu",
        noise_scale: float = 0.5,
        grad_clip: Optional[float] = 1.0,
        delta_action_clip: float = 0.5,
        add_grad_penalty: bool = True,
        grad_norm_type: str = "inf",
        grad_margin: float = 1.0,
        grad_loss_weight: float = 1.0,
        **kwargs,
    ):
        """Initialize the MCMC optimizer.

        Args:
            iters: Number of iterations
            use_langevin_negative_samples: Whether to use Langevin sampler
            train_samples: Number of samples for training
            inference_samples: Number of samples for inference
            stepsize_scheduler: Step size scheduler configuration
            optimize_again: Whether to run a second optimization
            again_stepsize_scheduler: Step size scheduler configuration for the second optimization
            device: Device to use for tensor operations
            noise_scale: Initial noise scale
            grad_clip: Gradient clipping value
            delta_action_clip: Action clipping value
            add_grad_penalty: Whether to add gradient penalty
            grad_norm_type: Gradient norm type ('1', '2', or 'inf')
            grad_margin: Gradient margin
            grad_loss_weight: Gradient loss weight
        """
        super().__init__(device)
        self.iters = iters
        self.use_langevin_negative_samples = use_langevin_negative_samples
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.stepsize_scheduler = stepsize_scheduler or {
            "init": 0.5,
            "final": 1e-5,
            "power": 2.0,
        }
        self.optimize_again = optimize_again
        self.again_stepsize_scheduler = again_stepsize_scheduler or {
            "init": 1e-5,
            "final": 1e-5,
            "power": 2.0,
        }
        self.noise_scale = noise_scale
        self.grad_clip = grad_clip
        self.delta_action_clip = delta_action_clip
        self.add_grad_penalty = add_grad_penalty
        self.grad_norm_type = grad_norm_type
        self.grad_margin = grad_margin
        self.grad_loss_weight = grad_loss_weight

    @staticmethod
    def _gradient_wrt_act(
        obs: torch.Tensor,
        action: torch.Tensor,
        ebm: nn.Module,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Calculate gradient with respect to action.

        Args:
            obs: Observations of shape (B, N, O)
            action: Actions of shape (B, N, A)
            ebm: Energy-based model
            create_graph: Whether to create computation graph for higher-order derivatives

        Returns:
            Gradient with respect to action of shape (B, N, A)
        """
        action.requires_grad_(True)
        energy = ebm.forward(obs, action)
        if energy.shape[-1] == 1 and len(energy.shape) > 1:
            energy = energy.squeeze(-1)
        energy = energy.sum()
        # `create_graph` set to `True` when second order derivative
        # is needed i.e, d(de/da)/d_param
        grad = torch.autograd.grad(energy, action, create_graph=create_graph)[0]
        action.requires_grad_(False)
        return grad

    def grad_penalty(
        self, obs: torch.Tensor, action: torch.Tensor, ebm: nn.Module
    ) -> torch.Tensor:
        """Calculate gradient penalty.

        Args:
            obs: Observations of shape (B, N+1, O)
            action: Actions of shape (B, N+1, A)
            ebm: Energy-based model

        Returns:
            Gradient penalty loss of shape (B,)
        """
        if not self.add_grad_penalty:
            return torch.tensor(0.0, device=self.device)

        # (B, N+1, A), this gradient is differentiable w.r.t model parameters
        de_dact = self._gradient_wrt_act(obs, action, ebm, create_graph=True)

        def compute_grad_norm(
            grad_norm_type: str, de_dact: torch.Tensor
        ) -> torch.Tensor:
            # de_deact: B, N+1, A
            # return:   B, N+1
            grad_norm_type_to_ord = {
                "1": 1,
                "2": 2,
                "inf": float("inf"),
            }
            ord = grad_norm_type_to_ord[grad_norm_type]
            return torch.linalg.norm(de_dact, ord, dim=-1)

        # (B, N+1)
        grad_norms = compute_grad_norm(self.grad_norm_type, de_dact)
        grad_norms = grad_norms - self.grad_margin
        grad_norms = grad_norms.clamp(min=0.0, max=1e10)
        grad_norms = grad_norms.pow(2)

        grad_loss = grad_norms.mean()
        return grad_loss * self.grad_loss_weight

    @no_ebm_grad()
    def _langevin_step(
        self, obs: torch.Tensor, action: torch.Tensor, stepsize: float, ebm: nn.Module
    ) -> torch.Tensor:
        """Run one Langevin MCMC step.

        Args:
            obs: Observations of shape (B, N, O)
            action: Actions of shape (B, N, A)
            stepsize: Step size
            ebm: Energy-based model

        Returns:
            Updated actions of shape (B, N, A)
        """
        l_lambda = 1.0
        de_dact = self._gradient_wrt_act(obs, action, ebm)

        if self.grad_clip is not None:
            de_dact = de_dact.clamp(min=-self.grad_clip, max=self.grad_clip)

        gradient_scale = 0.5
        de_dact = (
            gradient_scale * l_lambda * de_dact
            + torch.randn_like(de_dact) * l_lambda * self.noise_scale
        )

        delta_action = stepsize * de_dact
        delta_action_clip = (
            self.delta_action_clip
            * 0.5
            * (self.action_bounds[1] - self.action_bounds[0])
        )
        delta_action = delta_action.clamp(min=-delta_action_clip, max=delta_action_clip)

        action = action - delta_action
        action = action.clamp(min=self.action_bounds[0], max=self.action_bounds[1])

        return action

    @no_ebm_grad()
    def _langevin_action_given_obs(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        ebm: nn.Module,
        scheduler: Optional[BaseScheduler] = None,
    ) -> torch.Tensor:
        """Run Langevin MCMC for `self.iters` steps.

        Args:
            obs: Observations of shape (B, N, O)
            action: Actions of shape (B, N, A)
            ebm: Energy-based model
            scheduler: Learning rate scheduler

        Returns:
            Updated actions of shape (B, N, A)
        """
        if scheduler is None:
            self.stepsize_scheduler["num_steps"] = self.iters
            scheduler = self.PolynomialScheduler(**self.stepsize_scheduler)

        stepsize = scheduler.get_rate(-1)
        for i in range(self.iters):
            action = self._langevin_step(obs, action, stepsize, ebm)
            stepsize = scheduler.get_rate(i)
        return action

    @no_ebm_grad()
    def sample(
        self, obs: torch.Tensor, ebm: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create tiled observations and sample counter-negatives for InfoNCE loss.

        Args:
            obs: Observations of shape (B, O)
            ebm: Energy-based model

        Returns:
            A tuple containing:
                - tiled_obs: Tiled observations of shape (B, N, O)
                - action_samples: Action samples of shape (B, N, A)
        """
        obs, uniform_action_samples = self._sample(obs, self.train_samples)
        if not self.use_langevin_negative_samples:
            return obs, uniform_action_samples

        langevin_action_samples = self._langevin_action_given_obs(
            obs, uniform_action_samples, ebm
        )
        return obs, langevin_action_samples

    @no_ebm_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action using MCMC.

        Args:
            obs: Observations of shape (B, O)
            ebm: Energy-based model

        Returns:
            Best action samples of shape (B, A)
        """
        # (B, N, O), (B, N, A)
        obs, uniform_action_samples = self._sample(obs, self.inference_samples)
        action_samples = self._langevin_action_given_obs(
            obs,
            uniform_action_samples,
            ebm,
        )

        # Run a second optimization, a trick for more precise inference
        if self.optimize_again:
            self.again_stepsize_scheduler["num_steps"] = self.iters
            action_samples = self._langevin_action_given_obs(
                obs,
                action_samples,
                ebm,
                scheduler=self.PolynomialScheduler(**self.again_stepsize_scheduler),
            )

        # action_samples: B, N, A
        return self._get_best_action_sample(obs, action_samples, ebm)
