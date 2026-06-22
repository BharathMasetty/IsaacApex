# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""Kinematically replay a reference-motion ``.npz`` on the G1 for visual inspection.

Loads the maximal-coordinate motion produced by
[csv_to_npz.py](csv_to_npz.py) and drives the robot frame-by-frame (root pose +
joint angles written directly, no physics), looping forever. Useful for sanity-
checking a motion before training. Opens a GUI viewer by default; pass
``--headless`` to disable.

Example
-------
.. code-block:: bash

    python scripts/tracking/replay_npz.py \
        --motion_file source/IsaacApex/IsaacApex/tasks/tracking/motions/dance1_subject2.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import os

from isaaclab.app import AppLauncher

# Default to the motion bundled with the tracking task.
_DEFAULT_MOTION = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "source", "IsaacApex", "IsaacApex", "tasks", "tracking", "motions", "dance1_subject2.npz",
)

parser = argparse.ArgumentParser(description="Kinematically replay a reference motion .npz on the G1.")
parser.add_argument("--motion_file", type=str, default=os.path.abspath(_DEFAULT_MOTION), help="Path to the motion .npz.")
parser.add_argument("--loops", type=int, default=0, help="Number of loops to play (0 = forever).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from IsaacApex.assets.unitree_g1 import G1_29DOF_CFG


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    data = np.load(args_cli.motion_file)
    fps = int(data["fps"][0])
    joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=args_cli.device)
    joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=args_cli.device)
    # Root link (pelvis) is body index 0 in the recorded body arrays.
    root_pos = torch.tensor(data["body_pos_w"][:, 0], dtype=torch.float32, device=args_cli.device)
    root_quat = torch.tensor(data["body_quat_w"][:, 0], dtype=torch.float32, device=args_cli.device)
    root_lin_vel = torch.tensor(data["body_lin_vel_w"][:, 0], dtype=torch.float32, device=args_cli.device)
    root_ang_vel = torch.tensor(data["body_ang_vel_w"][:, 0], dtype=torch.float32, device=args_cli.device)
    n_frames = joint_pos.shape[0]
    print(f"[INFO]: Loaded {n_frames} frames @ {fps} fps ({n_frames / fps:.1f}s) from {args_cli.motion_file}")

    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / fps))
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    # Place camera to look at the motion.
    sim.set_camera_view(eye=[2.5, 2.5, 1.5], target=[0.0, 0.0, 0.8])
    print("[INFO]: Replaying (Ctrl-C to stop) ...")

    idx, loops = 0, 0
    while simulation_app.is_running():
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = root_pos[idx] + scene.env_origins
        root_state[:, 3:7] = root_quat[idx]
        root_state[:, 7:10] = root_lin_vel[idx]
        root_state[:, 10:13] = root_ang_vel[idx]
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos[idx : idx + 1], joint_vel[idx : idx + 1])

        sim.render()  # render only — kinematic playback, no physics
        scene.update(sim.get_physics_dt())

        idx += 1
        if idx >= n_frames:
            idx = 0
            loops += 1
            if args_cli.loops and loops >= args_cli.loops:
                break


if __name__ == "__main__":
    main()
    simulation_app.close()
