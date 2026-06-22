# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""Convert a LAFAN1-retargeted G1 CSV into a tracking-ready ``.npz`` motion.

The CSV (Unitree convention, e.g. from ``lvhaidong/LAFAN1_Retargeting_Dataset``)
holds per-row ``[base_pos(3), base_quat_xyzw(4), dof_pos(29)]``. This script
interpolates it to the control rate, computes base/joint velocities, drives the
G1 articulation in Isaac Sim by writing those states, and records the resulting
maximal-coordinate body states via forward kinematics. The output ``.npz`` has the
keys the :class:`MotionCommand` loader expects:

    fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

Unlike the upstream BeyondMimic script, this writes a plain local file (no
Weights & Biases registry).

Example
-------
.. code-block:: bash

    python scripts/tracking/csv_to_npz.py \
        --input_file source/IsaacApex/IsaacApex/assets/motions/dance1_subject2.csv \
        --input_fps 30 --frame_range 122 722 --output_fps 50 \
        --output_file source/IsaacApex/IsaacApex/tasks/tracking/motions/dance1_subject2.npz \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert a retargeted G1 CSV motion to a tracking .npz.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input motion CSV file.")
parser.add_argument("--input_fps", type=int, default=30, help="FPS of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Inclusive 1-based frame range to use. If omitted, the whole file is used.",
)
parser.add_argument("--output_file", type=str, required=True, help="Path to write the output .npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="FPS of the output motion (match the control rate).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from IsaacApex.assets.unitree_g1 import G1_29DOF_CFG, G1_29DOF_JOINTS


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    """Loads, interpolates, and differentiates a retargeted CSV motion."""

    def __init__(self, motion_file, input_fps, output_fps, device, frame_range):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
        self.motion_dof_poss_input = motion[:, 7:]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}); duration {self.duration:.2f}s, frames {self.input_frames}")

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0], self.motion_base_poss_input[index_1], blend.unsqueeze(1)
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0], self.motion_base_rots_input[index_1], blend
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0], self.motion_dof_poss_input[index_1], blend.unsqueeze(1)
        )
        print(f"Motion interpolated: {self.input_frames}@{self.input_fps} -> {self.output_frames}@{self.output_fps}")

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim, scene, joint_names):
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()  # kinematic replay — render only, no physics step
        scene.update(sim.get_physics_dt())

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[k] = np.stack(log[k], axis=0)
            out = os.path.abspath(args_cli.output_file)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out, **log)
            print(f"[INFO]: Saved motion ({log['joint_pos'].shape[0]} frames) to {out}")
            return  # main() returns -> __main__ closes the sim app once


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("[INFO]: Setup complete, replaying motion ...")
    run_simulator(sim, scene, joint_names=G1_29DOF_JOINTS)


if __name__ == "__main__":
    main()
    simulation_app.close()
