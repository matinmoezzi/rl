"""IBC Example.

This is a self-contained example of an offline IBC training script.

The helper functions are coded in the utils.py associated with this script.

"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from tensordict.nn import CudaGraphModule
from torchrl._utils import timeit
from torchrl.envs import set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    EBMPolicy,
    log_metrics,
    make_environment,
    make_ibc_model,
    make_ibc_optimizer,
    make_loss,
    make_offline_replay_buffer,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="", config_name="offline_config")
def main(cfg: DictConfig):  # noqa: F821
    set_gym_backend(cfg.env.backend).set()

    # Create logger
    exp_name = generate_exp_name("IBC-offline", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="ibc_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Creante env
    train_env, eval_env = make_environment(
        cfg,
        cfg.logger.eval_envs,
        logger=logger,
    )

    # Create replay buffer
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer)

    # Create agent
    model, stochastic_optimizer = make_ibc_model(cfg, train_env, eval_env, device)

    # Create loss
    loss_module = make_loss(model, stochastic_optimizer)

    # Create optimizer
    optimizer = make_ibc_optimizer(cfg.optim, loss_module)

    def update(data):
        optimizer.zero_grad(set_to_none=True)
        # compute losses
        loss_info = loss_module(data)
        loss = loss_info["loss"]

        loss.backward()
        optimizer.step()

        return loss_info.detach()

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    if cfg.compile.compile:
        update = torch.compile(update, mode=compile_mode)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, warmup=50)

    pbar = tqdm.tqdm(range(cfg.optim.gradient_steps))

    evaluation_interval = cfg.logger.eval_iter
    eval_steps = cfg.logger.eval_steps

    # Training loop
    for i in pbar:
        timeit.printevery(1000, cfg.optim.gradient_steps, erase=True)

        # sample data
        with timeit("sample"):
            data = replay_buffer.sample()
            data = data.to(device)

        with timeit("update"):
            torch.compiler.cudagraph_mark_step_begin()
            loss_info = update(data)

        # evaluation
        metrics_to_log = loss_info.to_dict()
        if (i + 1) % evaluation_interval == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), timeit("eval"):
                eval_td = eval_env.rollout(
                    max_steps=eval_steps,
                    policy=EBMPolicy(model, stochastic_optimizer),
                    auto_cast_to_device=True,
                )
                eval_env.apply(dump_video)
            eval_reward = eval_td["next", "reward"].sum(1).mean().item()
            metrics_to_log["evaluation_reward"] = eval_reward
            pbar.set_postfix({"eval_reward": eval_reward})
        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            if pbar.format_dict["rate"] is not None:
                metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, i)

        pbar.set_postfix({"loss": f"{loss_info['loss'].item():.4f}"})
    pbar.close()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()
