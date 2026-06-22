# Copyright (c) 2026 Bharath Masetty
# SPDX-License-Identifier: MIT
#
"""Side-by-side comparison of a trained tracking policy against the reference motion.

Spawns two G1s in the same scene:

* **policy robot** (at the env origin) — driven by the trained policy through the
  normal tracking env (full physics).
* **reference "ghost"** (offset in +y) — a second, gravity-free G1 driven
  *kinematically* straight from the motion command's reference pose each step.

Watching them step in lockstep shows exactly where the policy lags or drifts from
the target. Opens a GUI viewer by default; add ``--video`` to record instead.

Example
-------
.. code-block:: bash

    # auto-loads the latest g1_tracking checkpoint
    python scripts/tracking/compare_policy.py --num_envs 1

    # or a specific checkpoint
    python scripts/tracking/compare_policy.py --checkpoint <path/to/model_*.pt>
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare a tracking policy against the reference motion, side by side.")
parser.add_argument("--task", type=str, default="IsaacApex-G1-Tracking-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint (.pt). Default: latest g1_tracking.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of side-by-side pairs.")
parser.add_argument("--offset", type=float, default=1.5, help="Lateral (+y) gap between policy robot and reference.")
parser.add_argument("--loops", type=int, default=0, help="Full-motion passes to play before exiting (0 = forever).")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--real_time", action=argparse.BooleanOptionalAction, default=True, help="Play at wall-clock (real-time) speed."
)
parser.add_argument("--ghost_opacity", type=float, default=1.0, help="Reference-ghost opacity (0..1; 1=solid colour).")
parser.add_argument("--video", action="store_true", default=False, help="Record a video instead of opening a viewer.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# leave only hydra args on argv
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import importlib.metadata as importlib_metadata
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

from isaaclab_tasks.utils.hydra import hydra_task_config

import IsaacApex.tasks  # noqa: F401
from IsaacApex.assets.unitree_g1 import G1_29DOF_CFG

# rsl-rl changed its agent-config schema; the shims below adapt our cfg/checkpoint to the
# installed version (mirrors scripts/rsl_rl/play.py).
_RSL_RL_VERSION = importlib_metadata.version("rsl-rl-lib")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _find_latest_checkpoint() -> str | None:
    """Return the newest ``model_*.pt`` under the g1_tracking rsl_rl logs, if any."""
    root = os.path.join(_REPO, "scripts", "rsl_rl", "logs", "g1_tracking")
    for run in sorted(glob.glob(os.path.join(root, "*", "")), reverse=True):
        models = glob.glob(os.path.join(run, "model_*.pt"))
        if models:
            return max(models, key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
    return None


def _make_ghost_cfg(offset: float):
    """A gravity-free, drive-free clone of the G1 used as the kinematic reference."""
    ghost = copy.deepcopy(G1_29DOF_CFG)
    ghost.prim_path = "{ENV_REGEX_NS}/Ghost"
    ghost.spawn.activate_contact_sensors = False
    ghost.spawn.rigid_props.disable_gravity = True
    # No joint drives — the ghost's pose is written directly every step.
    for act in ghost.actuators.values():
        act.stiffness = 0.0
        act.damping = 0.0
    return ghost


def _shade_ghost(num_envs: int, opacity: float, color: tuple[float, float, float] = (0.80, 0.33, 0.00)):
    """Paint the reference ghost a vivid, self-lit colour so it is unmistakable next to the
    grey policy robot. Binds the material directly to every geometry prim with the strongest
    binding so it overrides the robot USD's own materials. Cosmetic — never aborts the run."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdShade

    mat_path = "/World/Looks/GhostMaterial"
    mat_cfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        emissive_color=color,  # self-lit so the colour reads regardless of scene lighting
        opacity=opacity,
        metallic=0.0,
        roughness=0.5,
    )
    mat_cfg.func(mat_path, mat_cfg)

    stage = omni.usd.get_context().get_stage()
    material = UsdShade.Material.Get(stage, mat_path)
    n_bound = 0
    for i in range(num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/Ghost")
        if not root.IsValid():
            continue
        # The robot's per-link "visuals" Xforms are instanceable, so their meshes are
        # instance proxies that can't be edited. De-instance them first so the binding
        # below actually reaches the geometry (otherwise it silently hits 0 meshes).
        for prim in Usd.PrimRange(root):
            if prim.IsInstanceable() and prim.GetName() == "visuals":
                prim.SetInstanceable(False)
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Gprim):  # mesh / capsule / etc. visual geometry
                UsdShade.MaterialBindingAPI(prim).Bind(
                    material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                )
                n_bound += 1
    print(f"[INFO] Shaded ghost: bound colour material to {n_bound} geometry prim(s).")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    device = args_cli.device if args_cli.device is not None else agent_cfg.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed

    # adapt the (old-style) agent cfg to the installed rsl-rl schema
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _RSL_RL_VERSION)

    # add the kinematic reference robot to the scene (picked up via cfg.__dict__)
    env_cfg.scene.ghost = _make_ghost_cfg(args_cli.offset)

    # clean, deterministic comparison: no domain randomisation, no obs noise, start on-reference
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.motion.debug_vis = False
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # Play the whole motion uninterrupted: drop the drift terminations and make the
    # episode effectively unbounded so the policy robot is never yanked mid-motion.
    # The motion command itself loops the timeline (see the frame-0 patch below).
    env_cfg.terminations.anchor_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None
    env_cfg.episode_length_s = 1.0e9

    # Isometric 3/4 view from a front corner, pulled back and elevated so both robots
    # are seen at an angle and the full body (incl. legs/feet) is in frame. The viewer
    # tracks the policy robot's root and is centred between the two robots (mid_y).
    mid_y = args_cli.offset / 2.0
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (4.0, mid_y - 4.0, 2.2)
    env_cfg.viewer.lookat = (0.0, mid_y, 0.3)

    resume_path = args_cli.checkpoint or _find_latest_checkpoint()
    if not resume_path or not os.path.isfile(resume_path):
        print(
            "[ERROR] No checkpoint found. Train first (scripts/rsl_rl/train.py) or pass --checkpoint <model_*.pt>.\n"
            f"        Looked under: {os.path.join(_REPO, 'scripts/rsl_rl/logs/g1_tracking')}"
        )
        return
    print(f"[INFO] Loading policy checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # motion length drives how many steps to record (one full pass per --loops, default 1)
    command = env.unwrapped.command_manager.get_term("motion")
    n_frames = int(command.motion.time_step_total)
    record_steps = max(args_cli.loops, 1) * n_frames

    if args_cli.video:
        video_folder = os.path.join(os.path.dirname(resume_path), "videos", "compare")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=record_steps,  # capture the full motion (× --loops)
            disable_logger=True,
        )
        print(f"[INFO] Recording {record_steps} steps ({record_steps / n_frames:.0f} full pass(es)) to: {video_folder}")

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, _RSL_RL_VERSION)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # handles to the reference ghost and the motion command
    ghost = env.unwrapped.scene["ghost"]
    _shade_ghost(env_cfg.scene.num_envs, args_cli.ghost_opacity)
    y_offset = torch.tensor([0.0, args_cli.offset, 0.0], device=env.unwrapped.device)
    zeros6 = torch.zeros(env.unwrapped.num_envs, 6, device=env.unwrapped.device)
    zeros_jv = torch.zeros_like(command.joint_pos)

    # Force the timeline to start at the first frame and loop sequentially: the env's
    # MotionCommand normally picks a random start frame (adaptive sampler) and the
    # motion-end resample picks another random one. Replacing the sampler with "go to
    # frame 0" makes both the initial reset and every loop restart begin the motion
    # from the top, so the full sequence is shown start-to-finish, over and over.
    def _start_at_zero(env_ids):
        command.time_steps[env_ids] = 0

    command._adaptive_sampling = _start_at_zero
    with torch.inference_mode():
        env.unwrapped.reset()  # apply the frame-0 start to both the policy robot (RSI) and the timeline

    def drive_ghost():
        # pelvis (tracked body 0) is the root link; render the reference pose statically
        root_pos = command.body_pos_w[:, 0] + y_offset
        root_state = torch.cat([root_pos, command.body_quat_w[:, 0], zeros6], dim=-1)
        ghost.write_root_state_to_sim(root_state)
        ghost.write_joint_state_to_sim(command.joint_pos, zeros_jv)

    obs = env.get_observations()
    step_dt = env.unwrapped.step_dt
    timestep, loops_done, prev_frame = 0, 0, -1
    while simulation_app.is_running():
        loop_start = time.time()
        with torch.inference_mode():
            frame = int(command.time_steps[0].item())
            if frame < prev_frame:  # timeline wrapped back to the start → one full pass done
                loops_done += 1
                print(f"[INFO] Completed motion pass {loops_done} ({n_frames} frames).")
                if args_cli.loops and loops_done >= args_cli.loops:
                    break
            prev_frame = frame

            drive_ghost()  # place the reference before stepping so it renders in sync
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)

            if args_cli.video:
                timestep += 1
                if timestep >= record_steps:
                    print(f"[INFO] Recorded {record_steps} steps. Saving video ...")
                    break

        # real-time pacing for the live viewer; skipped when recording (the saved
        # video's fps already encodes real-time playback).
        if args_cli.real_time and not args_cli.video:
            sleep_left = step_dt - (time.time() - loop_start)
            if sleep_left > 0:
                time.sleep(sleep_left)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
