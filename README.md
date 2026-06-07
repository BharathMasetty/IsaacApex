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

## Layout

```
IsaacApex/
├── source/IsaacApex/IsaacApex/
│   ├── assets/
│   │   └── unitree_g1.py       # G1 ArticulationCfg, joint lists, action scales
│   └── tasks/
│       └── locomotion/
│           ├── env_cfg.py      # G1LocomotionEnvCfg
│           ├── symmetry.py     # Left-right symmetry augmentation
│           ├── agents/
│           │   └── rsl_rl_cfg.py
│           └── mdp/
│               ├── commands.py
│               ├── observations.py
│               ├── rewards.py
│               ├── terminations.py
│               └── actions.py
└── scripts/
    ├── rsl_rl/
    │   ├── train.py
    │   └── play.py
    ├── list_envs.py
    ├── zero_agent.py
    └── random_agent.py
```
