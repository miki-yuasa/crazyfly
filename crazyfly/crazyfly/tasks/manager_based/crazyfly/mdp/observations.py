# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)

"""
Root state.
"""


@generic_io_descriptor(
    units="m",
    axes=["X", "Y", "Z"],
    observation_type="RootState",
    on_inspect=[record_shape, record_dtype],
)
def target_root_pos_w(
    env: ManagerBasedEnv,
    target_pos: tuple,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = asset.data.root_pos_w.shape[0]
    device = asset.data.root_pos_w.device
    dtype = asset.data.root_pos_w.dtype
    return torch.tensor(target_pos).to(device=device, dtype=dtype).repeat(num_envs, 1)
