# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

from utils.get_args import get_args, select_device


# add argparse arguments
parser = get_args()

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import crazyfly.tasks  # noqa: F401

from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.setup_logger import setup_logger
from utils.algorithms import ppo_policy

from utils.utils import get_traj_plot

# Add project root to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def evaluate(env, policy, env_idx=0):
    """Evaluate the policy for a number of episodes."""

    masks = torch.ones(NUM_ENVS, dtype=torch.bool, device=ENV_DEVICE)
    rendered_images = []
    x, y, z, rot1, rot2, rot3, rot4 = [], [], [], [], [], [], []

    # simulate environment
    G = 0
    count = 0
    with torch.inference_mode():
        obs, _ = env.reset()
        while simulation_app.is_running():
            actions, policy_info = policy(obs["policy"], deterministic=True)

            # step the environment
            next_obs, rew, terminated, truncated, env_info = env.step(actions)
            active_rew = rew[masks].mean()
            dones = terminated | truncated | (count >= EPISODE_LENGTH - 1)

            # render
            if RENDER:
                img = env.render()
                rendered_images.append(img)

            # logging
            G += (policy.gamma**count) * active_rew.item()
            count += 1

            if masks[env_idx]:
                x.append(obs["policy"][env_idx][3].cpu().numpy())
                y.append(obs["policy"][env_idx][4].cpu().numpy())
                z.append(obs["policy"][env_idx][5].cpu().numpy())
                rot1.append(obs["policy"][env_idx][16].cpu().numpy())
                rot2.append(obs["policy"][env_idx][17].cpu().numpy())
                rot3.append(obs["policy"][env_idx][18].cpu().numpy())
                rot4.append(obs["policy"][env_idx][19].cpu().numpy())

            if dones.any():
                masks[dones] = False

            if (~masks).all():
                break

            obs = next_obs

    fig = get_traj_plot(x, y, z, rot1, rot2, rot3, rot4)

    eval_info = {"Eval/return": G, "Eval/max_length": count}
    supp_info = {"Eval/fig": fig, "Eval/renderings": rendered_images}

    return eval_info, supp_info


def collect_samples(env, policy, deterministic=False):
    """Collect samples from the environment using the given policy.
    It should store in order as the environment is vectorized."""

    # do not use list but pre-allocate torch.nan for speed and debugging
    batches = dict(
        states=torch.full(
            (NUM_ENVS, EPISODE_LENGTH, STATE_DIM),
            fill_value=float("nan"),
            device=ENV_DEVICE,
        ),
        actions=torch.full(
            (NUM_ENVS, EPISODE_LENGTH, ACTION_DIM),
            fill_value=float("nan"),
            device=ENV_DEVICE,
        ),
        next_states=torch.full(
            (NUM_ENVS, EPISODE_LENGTH, STATE_DIM),
            fill_value=float("nan"),
            device=ENV_DEVICE,
        ),
        logprobs=torch.full(
            (NUM_ENVS, EPISODE_LENGTH, 1), fill_value=float("nan"), device=ENV_DEVICE
        ),
        rewards=torch.full(
            (NUM_ENVS, EPISODE_LENGTH, 1), fill_value=float("nan"), device=ENV_DEVICE
        ),
        terminals=torch.full(
            (NUM_ENVS, EPISODE_LENGTH, 1), fill_value=float("nan"), device=ENV_DEVICE
        ),
    )

    # simulate environment
    count = 0
    with torch.inference_mode():
        obs, _ = env.reset()
        while simulation_app.is_running():
            if count == 0:
                print(obs["policy"][:, :])
            actions, policy_info = policy(obs["policy"], deterministic)

            # step the environment
            next_obs, rew, terminated, truncated, env_info = env.step(actions)
            dones = terminated | truncated | (count >= EPISODE_LENGTH - 1)

            # logging
            batches["states"][:, count, :] = obs["policy"]
            batches["actions"][:, count, :] = actions
            batches["next_states"][:, count, :] = next_obs["policy"]
            batches["rewards"][:, count, :] = rew.unsqueeze(-1)
            batches["terminals"][:, count, :] = dones.unsqueeze(-1)
            batches["logprobs"][:, count, :] = policy_info["logprobs"]

            # break if all envs reach max episode length
            if count >= EPISODE_LENGTH - 1:
                break

            count += 1
            obs = next_obs

    # cat in dim 0 and cut nan values due to early termination
    batch = {k: v.reshape(-1, v.shape[-1]) for k, v in batches.items()}
    # check if nan exists and print warning
    if any(torch.isnan(v).any() for v in batch.values()):
        print("[WARNING]: NaN values found in batch and removed.")
        batch = {k: v[~torch.isnan(v).any(dim=1)] for k, v in batch.items()}
    # TODO: also render and send it to the wandb

    return batch


def main():
    """Zero actions agent with Isaac Lab environment."""
    # create environment
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # print(env_cfg.scene.camera.resolution, env_cfg.scene.camera.enable_cameras)
    # env_cfg.scene.camera.resolution = (1920, 1080)  # width, height
    # env_cfg.scene.camera.enable_cameras = True      # ensure cameras are enabled

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # ENV PARAMS that are global parameters #
    global NUM_ENVS, EPISODE_LENGTH, ENV_DEVICE, STATE_DIM, ACTION_DIM, RENDER
    NUM_ENVS = env_cfg.scene.num_envs
    EPISODE_LENGTH = int(env_cfg.episode_length_s // env_cfg.sim.dt)
    ENV_DEVICE = env_cfg.sim.device
    STATE_DIM = env.observation_space["policy"].shape[1]
    ACTION_DIM = env.action_space.shape[1]
    RENDER = True if args_cli.enable_cameras else False

    # print info (this is vectorized environment)
    # print(f"[INFO]: Gym observation space: {env.observation_space}")
    # print(f"[INFO]: Gym action space: {env.action_space}")

    # create policy with specifying device for training policy
    args_cli.device = select_device(args_cli.gpu_idx)
    policy = ppo_policy(args_cli, STATE_DIM, ACTION_DIM)

    # create logger
    logger = setup_logger(args_cli)

    # total number of timesteps to train
    eval_num = 0
    total_timesteps = args_cli.timesteps
    with tqdm(total=total_timesteps) as pbar:
        timesteps_done = 0
        while timesteps_done < total_timesteps and simulation_app.is_running():
            # collect a batch of samples from the environment
            batch = collect_samples(env, policy, deterministic=False)
            # update policy with collected batch
            loss_info, timesteps = policy.learn(batch)

            # evaluation
            if timesteps_done >= (args_cli.log_interval * eval_num):
                # evaluate the policy
                eval_info, supp_info = evaluate(env, policy)
                loss_info[f"Eval/{policy.name}/return"] = eval_info["Eval/return"]
                logger.write_image(
                    timesteps_done,
                    supp_info["Eval/fig"],
                    f"Eval/{policy.name}/trajectory",
                )
                if supp_info["Eval/renderings"]:
                    logger.write_videos(
                        timesteps_done,
                        supp_info["Eval/renderings"],
                        f"Eval/{policy.name}/renderings",
                    )
                eval_num += 1

            # log info
            logger.store(**loss_info)
            logger.write(timesteps_done, display=False)

            # increment timestep count
            timesteps_done += timesteps
            pbar.update(timesteps)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
