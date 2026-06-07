# IsaacApex

A research repository for implementing state-of-the-art reinforcement learning methods for **humanoid whole-body control**, built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab) and targeting the **Unitree G1** 29-DoF humanoid robot.

The goal is to provide clean, well-structured reference implementations of recent RL-for-locomotion and motion imitation papers — easy to read, easy to extend, and free of proprietary dependencies.

---

## What's Implemented

### 1. Velocity-Tracking Locomotion (`IsaacApex-G1-Locomotion-v0`)

A gait-aware locomotion policy that tracks commanded linear velocity, angular velocity, and root height using a periodic gait clock. Inspired by the reward structure of works like [Humanoid-Gym](https://arxiv.org/abs/2404.05695) and [OmniH2O](https://arxiv.org/abs/2406.08858).

**Key design choices:**
- **12-joint leg policy** — the policy controls only the 6+6 leg joints; waist and arms are held at default via a passive action term
- **Gait clock** — per-foot phase signals [sin, cos] × 2 drive gait-shaping rewards and are fed as observations, enabling the policy to learn emergent gaits without a reference motion
- **Subpopulation curriculum** — each episode is randomly assigned to standing (20%), turn-in-place (10%), heading-based (30%), or free walking, providing robust coverage of the command space
- **Asymmetric actor-critic** — the actor sees only proprioception (joint positions/velocities, IMU, gait phase); the critic additionally receives foot contact state, air time, contact forces, and foot heights
- **Left-right symmetry augmentation** — every training batch is doubled by mirroring observations and actions across the sagittal plane, improving sample efficiency and gate symmetry
- **DelayedPD actuators** — 0–4 physics-step command delay (0–20 ms) on all joints for realistic sim-to-real transfer

**Reward terms:** linear velocity tracking · angular velocity tracking · height tracking · gait phase synchronization · foot clearance · torso upright · default pose regularization · foot flat at contact · foot slip · foot impact velocity · action rate · joint velocity/torque/limit penalties · termination penalty

---

### 2. Motion Imitation (`IsaacApex-G1-MotionTracking-v0`)

A reference-motion tracking policy that imitates a motion capture clip, following the BeyondMimic / AMP family of approaches.

**Key design choices:**
- **Loads any .npz motion file** — the clip provides per-frame joint positions/velocities and body poses/velocities in world frame; the path is a configurable parameter
- **Adaptive start-frame sampling** — the clip is divided into bins; bins where episodes fail more often are sampled more frequently, pushing coverage toward hard regions of the motion
- **Full-body tracking** — rewards penalise deviation in anchor (pelvis) position and orientation, all body positions and orientations in the robot root frame, and body linear/angular velocities
- **Termination on large deviation** — episodes end if the anchor height or orientation, or any tracked body height, drifts beyond a threshold, keeping training signal meaningful
- **Asymmetric actor-critic** — actor receives proprioception + reference body poses relative to the robot root; critic additionally receives the robot's current body poses for privileged comparison

**Reward terms:** anchor position tracking · anchor orientation tracking · body position tracking (all links) · body orientation tracking · body linear velocity tracking · body angular velocity tracking · action rate · joint limit penalty · termination penalty

---

## Installation

**Prerequisites:** Isaac Lab installed and on your Python path. See the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
# Clone this repository outside the IsaacLab directory
git clone <this-repo-url> IsaacApex
cd IsaacApex

# Install in editable mode (use isaaclab.sh -p instead of python if not using conda/venv)
python -m pip install -e source/IsaacApex
```

Verify the install by listing registered environments:

```bash
python scripts/list_envs.py
```

You should see `IsaacApex-G1-Locomotion-v0` and `IsaacApex-G1-MotionTracking-v0`.

---

## Training

All training uses RSL-RL (PPO). Replace `python` with `isaaclab.sh -p` if Isaac Lab is not on your system Python path.

### Locomotion

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

### Motion Imitation

The motion file path must be set before training. Pass it as a config override:

```bash
python scripts/rsl_rl/train.py \
    --task IsaacApex-G1-MotionTracking-v0 \
    --num_envs 2048 \
    --headless \
    --motion_file /path/to/your/motion.npz
```

Or override it in a launch script:

```python
from IsaacApex.tasks.motion_tracking.env_cfg import G1MotionTrackingEnvCfg

env_cfg = G1MotionTrackingEnvCfg()
env_cfg.commands.motion_command.motion_file = "/path/to/your/motion.npz"
```

The `.npz` file must contain the following arrays (all `float32`):

| Key | Shape | Description |
|-----|-------|-------------|
| `fps` | scalar | Motion capture frame rate |
| `joint_names` | `(J,)` | Joint name strings |
| `body_names` | `(B,)` | Body name strings |
| `joint_pos` | `(T, J)` | Joint positions (rad) |
| `joint_vel` | `(T, J)` | Joint velocities (rad/s) |
| `body_pos_w` | `(T, B, 3)` | Body world positions (m) |
| `body_quat_w` | `(T, B, 4)` | Body world orientations (w, x, y, z) |
| `body_lin_vel_w` | `(T, B, 3)` | Body linear velocities (m/s) |
| `body_ang_vel_w` | `(T, B, 3)` | Body angular velocities (rad/s) |

---

## Playback / Evaluation

```bash
# Locomotion
python scripts/rsl_rl/play.py \
    --task IsaacApex-G1-Locomotion-v0 \
    --num_envs 16 \
    --load_run <run_name> \
    --checkpoint <checkpoint_file>

# Motion tracking
python scripts/rsl_rl/play.py \
    --task IsaacApex-G1-MotionTracking-v0 \
    --num_envs 4 \
    --load_run <run_name> \
    --checkpoint <checkpoint_file>
```

Sanity-check an environment with a zero-action or random agent (no training required):

```bash
python scripts/zero_agent.py --task IsaacApex-G1-Locomotion-v0
python scripts/random_agent.py --task IsaacApex-G1-Locomotion-v0
```

---

## Repository Layout

```
IsaacApex/
├── source/IsaacApex/IsaacApex/
│   ├── assets/
│   │   └── unitree_g1.py          # G1 29-DoF ArticulationCfg, joint lists, action scales
│   └── tasks/
│       ├── locomotion/
│       │   ├── env_cfg.py         # G1LocomotionEnvCfg
│       │   ├── symmetry.py        # Left-right symmetry augmentation
│       │   ├── agents/
│       │   │   └── rsl_rl_cfg.py  # G1LocomotionPPOCfg
│       │   └── mdp/
│       │       ├── commands.py    # GaitCommand — gait clock + subpopulations
│       │       ├── observations.py
│       │       ├── rewards.py
│       │       ├── terminations.py
│       │       └── actions.py     # HoldDefaultJointsAction
│       └── motion_tracking/
│           ├── env_cfg.py         # G1MotionTrackingEnvCfg
│           ├── agents/
│           │   └── rsl_rl_cfg.py  # G1MotionTrackingPPOCfg
│           └── mdp/
│               ├── commands.py    # MotionCommand — .npz loader + adaptive sampling
│               ├── observations.py
│               ├── rewards.py
│               └── terminations.py
└── scripts/
    ├── rsl_rl/
    │   ├── train.py
    │   └── play.py
    ├── list_envs.py
    ├── zero_agent.py
    └── random_agent.py
```

---

## Development Notes

**Adding a new task:** create a new subdirectory under `tasks/`, implement `env_cfg.py` and `mdp/`, register the gym environment in `__init__.py`, and add an `agents/rsl_rl_cfg.py`. The `import_packages` call in `tasks/__init__.py` will auto-discover it.

**Symmetry augmentation:** the locomotion policy's symmetry function is in `tasks/locomotion/symmetry.py`. Wire it into RSL-RL by setting the `symmetry_fn` field in the runner config or calling `locomotion_symmetry_fn` from a custom training loop.

**IDE setup:** press `Ctrl+Shift+P` → `Tasks: Run Task` → `setup_python_env` in VSCode to configure the Python environment for IntelliSense.
