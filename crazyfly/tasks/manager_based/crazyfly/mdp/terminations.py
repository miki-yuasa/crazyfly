# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_pos_out_of_bounds(env: ManagerBasedRLEnv, bounds:dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's root position is out of the specified bounds.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        bounds (dict[str, tuple[float, float]]): The (x, y) bounds for the root position.
        asset_cfg (SceneEntityCfg): Configuration for the asset to monitor.

    Returns:
        torch.Tensor: A boolean tensor indicating which environments should terminate.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    root_pos = asset.data.root_pos_w  # shape (num_envs, 3)

    x_out = (root_pos[:, 0] < bounds["x"][0]) | (root_pos[:, 0] > bounds["x"][1])
    y_out = (root_pos[:, 1] < bounds["y"][0]) | (root_pos[:, 1] > bounds["y"][1])

    # if torch.any(x_out | y_out):
    #     print(f"root_pos_out_of_bounds: x_out={x_out}, y_out={y_out}")

    return x_out | y_out
