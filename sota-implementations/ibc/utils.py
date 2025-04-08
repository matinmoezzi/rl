from __future__ import annotations

import functools

import numpy as np

import torch
import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDictBase

from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    RewardSum,
    TransformedEnv,
)

from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.modules import MLP
from torchrl.modules.planners.stochastic_optimizers import AutoRegressiveDFO, DFO, MCMC
from torchrl.objectives.ibc import IBCLoss
from torchrl.record import VideoRecorder
from torchrl.trainers.helpers.models import ACTIVATIONS


# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu", from_pixels=False):
    lib = cfg.env.backend
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
                categorical_action_encoding=True,
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=False
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def apply_env_transforms(
    env,
):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, train_num_envs=1, eval_num_envs=1, logger=None):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, cfg)
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env)

    maker = functools.partial(env_maker, cfg, from_pixels=cfg.logger.video)
    eval_env = TransformedEnv(
        ParallelEnv(
            eval_num_envs,
            EnvCreator(maker),
            serial_for_single=True,
        ),
        train_env.transform.clone(),
    )
    if cfg.logger.video:
        eval_env.insert_transform(
            0, VideoRecorder(logger, tag="rendered", in_keys=["pixels"])
        )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, compile_mode):
    """Make collector."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        init_random_frames=cfg.collector.init_random_frames,
        max_frames_per_traj=cfg.collector.max_frames_per_traj,
        total_frames=cfg.collector.total_frames,
        device=device,
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy=cfg.compile.cudagraphs,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


def make_offline_replay_buffer(rb_cfg):
    if rb_cfg.backend == "d4rl":
        data = D4RLExperienceReplay(
            dataset_id=rb_cfg.dataset,
            split_trajs=False,
            batch_size=rb_cfg.batch_size,
            # We use drop_last to avoid recompiles (and dynamic shapes)
            sampler=SamplerWithoutReplacement(drop_last=True),
            prefetch=4,
            direct_download=True,
        )
    elif rb_cfg.backend == "minari":
        data = MinariExperienceReplay(
            dataset_id=rb_cfg.dataset,
            split_trajs=False,
            batch_size=rb_cfg.batch_size,
            sampler=SamplerWithoutReplacement(drop_last=True),
            prefetch=4,
            download=True,
        )
    else:
        raise NotImplementedError(f"Unknown backend {rb_cfg.backend}.")

    data.append_transform(DoubleToFloat())

    return data


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#


class EBMPolicy(TensorDictModule):
    """Policy module that uses EBM network and stochastic optimizer for inference.

    This module takes observations as input and returns the best action using
    the stochastic optimizer's inference method.

    Args:
        ebm_network (TensorDictModule): The energy-based model network
        optimizer (Union[MCMC, AutoRegressiveDFO, DFO]): The stochastic optimizer
    """

    def __init__(
        self,
        ebm_network: TensorDictModule,
        optimizer: MCMC | AutoRegressiveDFO | DFO,
    ):
        super().__init__(
            module=self,
            in_keys=["observation"],
            out_keys=["action"],
        )
        self.ebm_network = ebm_network
        self.optimizer = optimizer

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Forward pass that infers the best action for given observations.

        Args:
            tensordict (TensorDictBase): Input tensordict containing observations

        Returns:
            TensorDictBase: Tensordict with inferred actions
        """
        obs = tensordict.get("observation")
        # Use optimizer's inference method to get best actions
        actions = self.optimizer.infer(obs, self.ebm_network)
        tensordict.set("action", actions)
        return tensordict


def make_ibc_model(cfg, train_env, eval_env, device="cpu"):
    model_cfg = cfg.model

    action_spec = train_env.action_spec_unbatched

    # Create EBM network
    ebm_net = MLP(
        num_cells=model_cfg.num_cells,
        depth=model_cfg.depth,
        out_features=1,  # Energy value
        activation_class=ACTIVATIONS[model_cfg.activation],
        device=device,
    )

    # Create EBM module
    ebm_module = TensorDictModule(
        module=ebm_net,
        in_keys=["observation", "action"],
        out_keys=["energy"],
    )

    # Create optimizer based on config
    optimizer_cfg = model_cfg.optimizer
    if optimizer_cfg.type == "dfo":
        optimizer = DFO(device=device)
    elif optimizer_cfg.type == "mcmc":
        optimizer = MCMC(device=device)
    elif optimizer_cfg.type == "autoregressive_dfo":
        optimizer = AutoRegressiveDFO(device=device)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_cfg.type}")

    # Set action bounds from environment spec
    if hasattr(action_spec.space, "low") and hasattr(action_spec.space, "high"):
        action_bounds = np.stack(
            [action_spec.space.low, action_spec.space.high], axis=0
        )
        optimizer.set_action_bounds(action_bounds)
    else:
        raise ValueError("Action spec must have low and high bounds for the optimizer")

    # Initialize the model
    with torch.no_grad():
        td = eval_env.fake_tensordict()
        td = td.to(device)
        ebm_module(td)

    return ebm_module, optimizer


# ====================================================================
# IBC Loss
# ---------


def make_loss(model, stochastic_optimizer):
    loss_module = IBCLoss(model, stochastic_optimizer)
    return loss_module


def make_ibc_optimizer(optim_cfg, loss_module):
    optimizer = torch.optim.AdamW(
        loss_module.parameters(),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
        betas=optim_cfg.get("betas", (0.9, 0.999)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=optim_cfg.lr_anneal_init,
        T_mult=optim_cfg.lr_anneal_mult,
        eta_min=optim_cfg.lr_anneal_min,
    )
    return optimizer, scheduler


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    if logger is not None:
        for metric_name, metric_value in metrics.items():
            logger.log_scalar(metric_name, metric_value, step)


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
