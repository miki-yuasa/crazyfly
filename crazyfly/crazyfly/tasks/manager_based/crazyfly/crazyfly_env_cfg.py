# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG  # isort:skip
from isaaclab.markers.config import SPHERE_MARKER_CFG


##
# Scene definition
##


@configclass
class CrazyflySceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane with bright color
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0), color=(1.0, 1.0, 0.8)  # bright yellowish
        ),
    )

    # robot colored red or blue
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        debug_vis=True,
    )

    # # Example: spawning a red cone at (-1, 1, 1)
    # cone = AssetBaseCfg(
    #     prim_path="/World/Objects/Cone",
    #     spawn=sim_utils.ConeCfg(
    #         radius=0.15,
    #         height=0.5,
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(1.0, 0.0, 0.0),
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=TARGET_POS,            # ✅ position here
    #         rot=(1.0, 0.0, 0.0, 0.0),  # ✅ quaternion (w, x, y, z)
    #     ),
    # )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # motor_efforts = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=["m1_joint", "m2_joint", "m3_joint", "m4_joint"],
    #     scale=1e-0,   # tune this so actions map to realistic thrust
    # )

    control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(use_motor_model=False)

    # control_action = mdp.JointEffortActionCfg(asset_name="robot",joint_names=["m1_joint", "m2_joint", "m3_joint", "m4_joint"],use_motor_model=False)


@configclass
class ObservationsCfg:
    """Observation specifications for the Crazyflie MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Drone’s base pose
        target_root_pos = ObsTerm(
            func=mdp.target_root_pos_w, params={"target_pos": (0, 0, 3)}
        )  # target position (x, y, z)
        root_pos = ObsTerm(func=mdp.root_pos_w)  # world-frame position (x, y, z)
        root_rot = ObsTerm(func=mdp.root_quat_w)  # orientation quaternion

        # Drone’s base velocities
        root_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w
        )  # linear velocity in world frame
        root_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w
        )  # angular velocity in world frame

        # Motor states (relative joint velocity)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # pose_command = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "pose_command"}
        # )

        def __post_init__(self) -> None:
            # Ensures that, no matter what defaults or inputs were provided, the config will disable corruption (noise/perturbations) after initialization.
            self.enable_corruption = False
            # Forces the config to concatenate different observation components (like position, velocity, orientation) into a single vector.
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events for the Crazyflie quadcopter."""

    reset_root = EventTerm(
        func=mdp.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["m1_joint", "m2_joint", "m3_joint", "m4_joint"]
            ),
            "pose_range": dict(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.3, 0.5)),
            "velocity_range": dict(
                x=(-0.1, 0.1),
                y=(-0.1, 0.1),
                z=(-0.1, 0.1),
                roll=(-0.1, 0.1),
                pitch=(-0.1, 0.1),
                yaw=(-0.1, 0.1),
            ),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP (hovering Crazyflie)."""

    # (1) Constant reward for staying alive
    alive = RewTerm(func=mdp.is_alive, weight=0.1)

    # (2) Failure penalty (crash or out-of-bounds)
    terminating = RewTerm(func=mdp.is_terminated, weight=-100.0)

    # (3) Primary task: hover at target position
    hover_pos = RewTerm(
        func=mdp.base_height_l2,  # L2 distance to target position
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 3.0,
        },  # hover at z=3.0
    )

    # (4) Velocity shaping: minimize linear velocity
    lin_vel_xy = RewTerm(
        func=mdp.lin_vel_xy_l2,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (5) Velocity shaping: minimize angular velocity
    ang_vel_z = RewTerm(
        func=mdp.ang_vel_z_l2,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# @configclass
# class CommandsCfg:
#     """Command terms for the MDP."""

#     pose_command = mdp.UniformPoseCommandCfg(
#         asset_name="robot",  # must match your CrazyflySceneCfg.robot name
#         body_name="body",  # must match body in your robot USD
#         resampling_time_range=(1.0e9, 1.0e9),  # no resampling in the episode
#         debug_vis=True,
#         ranges=mdp.UniformPoseCommandCfg.Ranges(
#             pos_x=(-0.1, 0.1),  # x ∈ [-0.1, 0.1]
#             pos_y=(-0.1, 0.1),  # y ∈ [-0.1, 0.1]
#             pos_z=(0.9, 1.1),  # z ∈ [0.9, 1.1]
#             roll=(0.0, 0.0),  # fixed roll
#             pitch=(0.0, 0.0),  # fixed pitch
#             yaw=(-3.14, 3.14),  # full yaw rotation
#         ),
#         goal_pose_visualizer_cfg=SPHERE_MARKER_CFG.replace(
#             prim_path="/Visuals/Command/goal_pose"
#         ),
#     )


@configclass
class TerminationsCfg:
    """Termination conditions for the MDP (hovering Crazyflie)."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Out-of-bounds position (X, Y)
    xy_out_of_bounds = DoneTerm(
        func=mdp.root_pos_out_of_bounds,  # checks if joint positions exceed soft joint limits
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bounds": dict(x=(-10.0, 10.0), y=(-10.0, 10.0)),
        },
    )

    # (3) Tilt limits (roll/pitch)
    tilt_out_of_bounds = DoneTerm(
        func=mdp.bad_orientation,  # terminates if tilt exceeds limit
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": 3.14,  # radians; approximately 85.9 degrees
        },
    )

    # (4) Root height below minimum
    root_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": 0.1,  # minimum height above ground
        },
    )


##
# Environment configuration
##


@configclass
class CrazyflyEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: CrazyflySceneCfg = CrazyflySceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    # commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
