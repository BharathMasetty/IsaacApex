# IsaacApex

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reinforcement learning for **humanoid whole-body control**, built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab) and targeting the **Unitree G1** 29-DoF humanoid robot.

Clean reference implementations of recent RL-for-locomotion methods — easy to read, easy to extend, no proprietary dependencies.

---

## Installation

**Prerequisites:** Isaac Lab installed and on your Python path. See the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
git clone <this-repo-url> IsaacApex
cd IsaacApex
python -m pip install -e source/IsaacApex
```

Verify:

```bash
python scripts/list_envs.py
```

---

## Training

All training uses RSL-RL (PPO). Replace `python` with `isaaclab.sh -p` if Isaac Lab is not on your system Python path.

```bash
python scripts/rsl_rl/train.py \
    --task IsaacApex-G1-Locomotion-v0 \
    --num_envs 4096 \
    --headless
```

Resume from a checkpoint:

```bash
python scripts/rsl_rl/train.py \
    --task IsaacApex-G1-Locomotion-v0 \
    --num_envs 4096 \
    --headless \
    --resume \
    --load_run <run_name> \
    --checkpoint <checkpoint_file>
```

---

## Evaluation

```bash
python scripts/rsl_rl/play.py \
    --task IsaacApex-G1-Locomotion-v0 \
    --num_envs 16 \
    --load_run <run_name> \
    --checkpoint <checkpoint_file>
```

Zero/random agent sanity checks (no training required):

```bash
python scripts/zero_agent.py --task IsaacApex-G1-Locomotion-v0
python scripts/random_agent.py --task IsaacApex-G1-Locomotion-v0
```

---

## Motion Tracking (BeyondMimic-style)

`IsaacApex-G1-Tracking-v0` trains the **29-DoF** G1 to imitate a reference motion —
by default an expressive **LAFAN1 dance** (`dance1_subject2`) retargeted to the G1.
The task is a port of [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking):
DeepMimic-style per-body tracking rewards, anchor-relative observations, reference-state
initialisation with an adaptive start-frame sampler, and domain randomisation.

The reference motion ships in the repo
(`source/IsaacApex/IsaacApex/tasks/tracking/motions/dance1_subject2.npz`), so
training runs with no extra setup:

```bash
python scripts/rsl_rl/train.py \
    --task IsaacApex-G1-Tracking-v0 \
    --num_envs 4096 \
    --headless
```

Play a trained checkpoint (the reference skeleton is drawn alongside the robot):

```bash
python scripts/rsl_rl/play.py \
    --task IsaacApex-G1-Tracking-v0 \
    --num_envs 4 \
    --load_run <run_name> --checkpoint <checkpoint_file>
```

### How the environment works

Defined in [env_cfg.py](source/IsaacApex/IsaacApex/tasks/tracking/env_cfg.py)
(`G1TrackingEnvCfg`); the reference-motion command lives in
[mdp/commands.py](source/IsaacApex/IsaacApex/tasks/tracking/mdp/commands.py).

| Aspect | Detail |
|---|---|
| **Robot / action** | 29-DoF G1 (6+6 legs, 3 waist, 7+7 arms); joint-position targets. The USD's finger joints are not actuated — held folded by the `hands` actuator. |
| **Control rate** | 50 Hz policy (`decimation=4`) over 200 Hz physics (`sim.dt=0.005`); 20 s episodes. |
| **Tracked bodies** | 14 body frames; **torso** is the anchor (root reference frame); the 4 end-effectors are the ankles and wrists. |
| **Observations** | *policy*: reference joint targets + anchor-relative pose error + proprioception (with observation noise); *critic*: the same, noise-free, plus privileged per-body frames (asymmetric actor-critic). |
| **Rewards** | Exp-kernel tracking errors on the anchor pose (pos/ori) and per-body pose + velocity (pos/ori/lin-vel/ang-vel), DeepMimic-style, minus action-rate, joint-limit, and undesired-contact penalties. |
| **Terminations** | 20 s timeout, anchor height/orientation drift, and end-effector height drift. |
| **Reset (RSI)** | Reference State Initialisation — teleport to a sampled reference frame with small pose/velocity/joint perturbations. An **adaptive sampler** revisits the motion phases the policy fails on most, acting as a built-in curriculum. |
| **Domain rand.** | Per-body friction/restitution and torso centre-of-mass (startup), plus random base-velocity pushes every 1–3 s. |

Training uses RSL-RL PPO with an asymmetric actor-critic (the privileged `critic`
observation group), empirical observation normalisation, and an adaptive-KL
learning-rate schedule — see
[agents/rsl_rl_cfg.py](source/IsaacApex/IsaacApex/tasks/tracking/agents/rsl_rl_cfg.py).

### Inspecting a motion

Kinematically replay any tracking `.npz` on the G1 (root pose + joint angles
written directly, no policy, no physics) to sanity-check it before training. Opens
a GUI viewer by default; pass `--headless` to disable:

```bash
python scripts/tracking/replay_npz.py \
    --motion_file source/IsaacApex/IsaacApex/tasks/tracking/motions/dance1_subject2.npz
```

### Using a different motion

Grab any G1-retargeted CSV from the
[LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
(`[base_pos(3), base_quat_xyzw(4), dof_pos(29)]` per row) and convert it to a
tracking `.npz` via forward kinematics in Isaac Sim. The converter interpolates to
the control rate, differentiates for velocities, and writes the keys the loader
expects (`fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w,
body_ang_vel_w`) — no Weights & Biases registry, just a plain local file:

```bash
python scripts/tracking/csv_to_npz.py \
    --input_file source/IsaacApex/IsaacApex/assets/motions/dance1_subject2.csv \
    --input_fps 30 --frame_range 122 722 --output_fps 50 \
    --output_file source/IsaacApex/IsaacApex/tasks/tracking/motions/dance1_subject2.npz \
    --headless
```

`--output_fps` must match the control rate (50 Hz); `--frame_range` is an inclusive
1-based clip of the input (omit it to use the whole file). Point the task at a new
motion by editing `DEFAULT_MOTION_FILE` in
[env_cfg.py](source/IsaacApex/IsaacApex/tasks/tracking/env_cfg.py) (or overriding
`commands.motion.motion_file`).

---

## Layout

```
IsaacApex/
├── source/IsaacApex/IsaacApex/
│   ├── assets/
│   │   ├── unitree_g1.py       # G1 cfgs: 37-DoF USD (locomotion) + 29-DoF USD (tracking)
│   │   └── motions/            # bundled retargeted source CSV(s)
│   └── tasks/
│       ├── locomotion/
│       │   ├── env_cfg.py      # G1LocomotionEnvCfg
│       │   ├── symmetry.py     # Left-right symmetry augmentation
│       │   ├── agents/         # rsl_rl_cfg.py
│       │   └── mdp/            # commands, observations, rewards, terminations, actions
│       └── tracking/           # BeyondMimic-style motion tracking
│           ├── env_cfg.py      # G1TrackingEnvCfg
│           ├── motions/        # bundled reference .npz files
│           ├── agents/         # rsl_rl_cfg.py
│           └── mdp/            # commands (MotionCommand), observations, rewards, terminations, events
└── scripts/
    ├── rsl_rl/                 # train.py, play.py (PPO)
    ├── tracking/
    │   ├── csv_to_npz.py       # LAFAN1 CSV -> tracking .npz (no W&B needed)
    │   └── replay_npz.py       # kinematic replay of a tracking .npz for inspection
    ├── list_envs.py
    ├── zero_agent.py
    └── random_agent.py
```
