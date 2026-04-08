import argparse
import json
from copy import deepcopy

import torch


def get_args():
    parser = argparse.ArgumentParser(description="")

    # === WandB parameters === #
    parser.add_argument(
        "--project", type=str, default="Exp", help="WandB project classification"
    )

    # === Isaac Lab parameters === #
    parser.add_argument(
        "--task", type=str, default="Isaac-Crazyfly-v0", help="Name of the task."
    )
    parser.add_argument(
        "--num_envs", type=int, default=3, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )

    # === Experiment parameters === #
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs.")

    # === Algorithmic parameters === #
    parser.add_argument(
        "--algo_name", type=str, default="ppo", help="Name of the algorithm to run."
    )
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=5e-5,
        help="Base learning rate for the actor for baselines.",
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=1e-4,
        help="Base learning rate for the critic for all algorithms.",
    )
    parser.add_argument(
        "--eps_clip", type=float, default=0.2, help="PPO clipping parameter."
    )
    parser.add_argument(
        "--actor_fc_dim",
        type=list,
        default=[256, 256],
        help="actor fc layer dimensions.",
    )
    parser.add_argument(
        "--critic_fc_dim",
        type=list,
        default=[256, 256],
        help="critic fc layer dimensions.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1e6, help="Number of training timesteps."
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1_000_000,
        help="Interval for logging results.",
    )
    parser.add_argument("--K_epochs", type=int, default=10, help="")
    parser.add_argument("--num_minibatch", type=int, default=5, help="")
    parser.add_argument(
        "--target_kl",
        type=float,
        default=1e-2,
        help="Target KL constraint.",
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="Generalized Advantage Estimation (GAE) factor.",
    )
    parser.add_argument(
        "--entropy_scaler", type=float, default=1e-3, help="Base learning rate."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--gpu_idx", type=int, default=0, help="Index of the GPU to use."
    )

    return parser


def select_device(gpu_idx=0, verbose=False):
    if verbose:
        print(
            "============================================================================================"
        )
        # set device to cpu or cuda
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
    return device
