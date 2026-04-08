# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

# from dynamics import Allocation, Motor
# from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class Allocation:
    def __init__(self, num_envs, arm_length, thrust_coeff, drag_coeff, device="cpu", dtype=torch.float32):
        """
        Initializes the allocation matrix for a quadrotor for multiple environments.

        Parameters:
        - num_envs (int): Number of environments
        - arm_length (float): Distance from the center to the rotor
        - thrust_coeff (float): Rotor thrust constant
        - drag_coeff (float): Rotor torque constant
        - device (str): 'cpu' or 'cuda'
        - dtype (torch.dtype): Desired tensor dtype
        """
        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        A = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [arm_length * sqrt2_inv, -arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [-arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [drag_coeff, -drag_coeff, drag_coeff, -drag_coeff],
            ],
            dtype=dtype,
            device=device,
        )
        self._allocation_matrix = A.unsqueeze(0).repeat(num_envs, 1, 1)
        self._thrust_coeff = thrust_coeff

    def compute(self, omega):
        """
        Computes the total thrust and body torques given the rotor angular velocities.

        Parameters:
        - omega (torch.Tensor): Tensor of shape (num_envs, 4) representing rotor angular velocities

        Returns:
        - thrust_torque (torch.Tensor): Tensor of shape (num_envs, 4)
        """
        thrusts_ref = self._thrust_coeff * omega**2
        thrust_torque = torch.bmm(self._allocation_matrix, thrusts_ref.unsqueeze(-1)).squeeze(-1)
        return thrust_torque


class Motor:
    def __init__(self, num_envs, taus, init, max_rate, min_rate, dt, use, device="cpu", dtype=torch.float32):
        """
        Initializes the motor model.

        Parameters:
        - num_envs: Number of envs.
        - taus: (4,) Tensor or list specifying time constants per motor.
        - init: (4,) Tensor or list specifying the initial omega per motor. (rad/s)
        - max_rate: (4,) Tensor or list specifying max rate of change of omega per motor. (rad/s^2)
        - min_rate: (4,) Tensor or list specifying min rate of change of omega per motor. (rad/s^2)
        - dt: Time step for integration.
        - use: Boolean indicating whether to use motor dynamics.
        - device: 'cpu' or 'cuda' for tensor operations.
        - dtype: Data type for tensors.
        """
        self.num_envs = num_envs
        self.num_motors = len(taus)
        self.dt = dt
        self.use = use
        self.init = init
        self.device = device
        self.dtype = dtype

        self.omega = torch.tensor(init, device=device).expand(num_envs, -1).clone()  # (num_envs, num_motors)

        # Convert to tensors and expand for all drones
        self.tau = torch.tensor(taus, device=device).expand(num_envs, -1)  # (num_envs, num_motors)
        self.max_rate = torch.tensor(max_rate, device=device).expand(num_envs, -1)  # (num_envs, num_motors)
        self.min_rate = torch.tensor(min_rate, device=device).expand(num_envs, -1)  # (num_envs, num_motors)

    def compute(self, omega_ref):
        """
        Computes the new omega values based on reference omega and motor dynamics.

        Parameters:
        - omega_ref: (num_envs, num_motors) Tensor of reference omega values.

        Returns:
        - omega: (num_envs, num_motors) Tensor of updated omega values.
        """

        if not self.use:
            self.omega = omega_ref
            return self.omega

        # Compute omega rate using first-order motor dynamics
        omega_rate = (1.0 / self.tau) * (omega_ref - self.omega)  # (num_envs, num_motors)
        omega_rate = omega_rate.clamp(self.min_rate, self.max_rate)

        # Integrate
        self.omega += self.dt * omega_rate
        return self.omega

    def reset(self, env_ids):
        """
        Resets the motor model to initial conditions.
        """
        self.omega[env_ids] = torch.tensor(self.init, device=self.device, dtype=self.dtype).expand(len(env_ids), -1)


class ControlAction(ActionTerm):
    r"""Body torque control action term.

    This action term applies a wrench to the drone body frame based on action commands

    """

    cfg: ControlActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ControlActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._allocation = Allocation(
            num_envs=self.num_envs,
            arm_length=self.cfg.arm_length,
            thrust_coeff=self.cfg.thrust_coef,
            drag_coeff=self.cfg.drag_coef,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._motor = Motor(
            num_envs=self.num_envs,
            taus=self.cfg.taus,
            init=self.cfg.init,
            max_rate=self.cfg.max_rate,
            min_rate=self.cfg.min_rate,
            dt=env.physics_dt,
            use=self.cfg.use_motor_model,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # TODO: make more explicit (thrust = 6, rates = 6, attitude = 6) all happen to be 6, but they represent different things
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # normalize the actions for RL-friendly range 
        clamped = self._raw_actions.clamp_(-1.0, 1.0)
        mapped = (clamped + 1.0) / 2.0
        # compute the motor angular velocities
        omega_ref = self.cfg.omega_max * mapped
        omega_real = self._motor.compute(omega_ref)
        self._processed_actions = self._allocation.compute(omega_real)

    def apply_actions(self):
        # self._thrust[:, 0, 2] = self._processed_actions[:, 0]
        # self._moment[:, 0, :] = self._processed_actions[:, 1:]
        # self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)

        # self._elapsed_time += self._env.physics_dt

    def reset(self, env_ids):
        # if env_ids is None or len(env_ids) == self.num_envs:
            # env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        # self._processed_actions[env_ids] = 0.0
        # self._elapsed_time[env_ids] = 0.0

        # self._motor.reset(env_ids)
        # self._robot.reset(env_ids)
        # joint_pos = self._robot.data.default_joint_pos[env_ids]
        # joint_vel = self._robot.data.default_joint_vel[env_ids]
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._env.scene.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@configclass
class ControlActionCfg(ActionTermCfg):
    """
    See :class:`ControlAction` for more details.
    """

    class_type: type[ActionTerm] = ControlAction
    """ Class of the action term."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
    arm_length: float = 0.035
    """Length of the arms of the drone in meters."""
    drag_coef: float = 1.5e-9
    """Drag torque coefficient."""
    thrust_coef: float = 2.25e-7
    """Thrust coefficient.
    Calculated with 5145 rad/s max angular velociy, thrust to weight: 4, mass: 0.6076 kg and gravity: 9.81 m/s^2.
    thrust_coef = (4 * 0.6076 * 9.81) / (4 * 5145**2) = 2.25e-7."""
    omega_max: float = 5145.0
    """Maximum angular velocity of the drone motors in rad/s.
    Calculated with 1950KV motor, with 6S LiPo battery with 4.2V per cell.
    1950 * 6 * 4.2 = 49,140 RPM ~= 5145 rad/s."""
    taus: list[float] = (0.0001, 0.0001, 0.0001, 0.0001)
    """Time constants for each motor."""
    init: list[float] = (2572.5, 2572.5, 2572.5, 2572.5)
    """Initial angular velocities for each motor in rad/s."""
    max_rate: list[float] = (50000.0, 50000.0, 50000.0, 50000.0)
    """Maximum rate of change of angular velocities for each motor in rad/s^2."""
    min_rate: list[float] = (-50000.0, -50000.0, -50000.0, -50000.0)
    """Minimum rate of change of angular velocities for each motor in rad/s^2."""
    use_motor_model: bool = False
    """Flag to determine if motor delay is bypassed."""
