# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Reinforcement learning environments for Unitree robots, built on [IsaacLab](https://github.com/isaac-sim/IsaacLab).

> **Fork** of [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) with custom policies and motion capture pipelines.

<div align="center">

| Isaac Lab | MuJoCo | Physical |
|---|---|---|
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

---

## Supported Robots

| Robot | Locomotion | Mimic (Dance) |
|-------|------------|---------------|
| **G1-29dof** | ✅ | ✅ |
| **G1-23dof** | ✅ | ✅ |
| **Go2** | ✅ | ❌ |
| **H1** | ✅ | ❌ |

---

## Pre-trained Policies

Ready-to-use policies in `deploy/robots/`:

### G1-29dof

| Policy | Type | Description | Gamepad |
|--------|------|-------------|---------|
| **Velocity** | Locomotion | Walking with velocity control | `RB + X` |
| **Gangnam Style** | Mimic | Dance motion | `LT + ←` |
| **Dance 102** | Mimic | Dance motion | `LT + ↓` |
| **My Dance** | Mimic | Custom dance | `LT + →` |
| **Bata Dias** | Mimic | Rokoko motion capture | `LT + A` |

### G1-23dof

| Policy | Type | Description |
|--------|------|-------------|
| **Velocity** | Locomotion | Walking |
| **Bata Dias** | Mimic | Rokoko motion capture |

---

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/diaskabdualiev/unitree_rl_lab.git
cd unitree_rl_lab

# Install (requires Isaac Lab)
conda activate env_isaaclab
./unitree_rl_lab.sh -i
```

### Training

```bash
# Locomotion (walking)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless

# Mimic (dance)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Mimic-Bata-Dias --headless
```

### Inference

```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity
```

---

## Deploy to Real Robot / MuJoCo

### Build Controller

```bash
# Install unitree_sdk2
git clone https://github.com/unitreerobotics/unitree_sdk2
cd unitree_sdk2 && mkdir build && cd build
cmake .. && sudo make install

# Build g1_ctrl
cd unitree_rl_lab/deploy/robots/g1_29dof
mkdir build && cd build
cmake .. && make
```

### Run in MuJoCo (Simulation)

```bash
# Terminal 1: MuJoCo
cd unitree_mujoco/simulate/build
./unitree_mujoco

# Terminal 2: Controller
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network lo
```

### Run on Real Robot

```bash
./g1_ctrl --network eth0
```

### Gamepad Controls

| Combo | Action |
|-------|--------|
| `LT + ↑` | Stand up (FixStand) |
| `RB + X` | Walk (Velocity) |
| `LT + ↓` | Dance 102 |
| `LT + ←` | Gangnam Style |
| `LT + →` | My Dance |
| `LT + A` | Bata Dias |
| `LT + B` | Stop (Passive) |
| Left Stick | Move |
| Right Stick | Rotate |

---

## Available Tasks

```bash
./unitree_rl_lab.sh -l
```

### Locomotion

| Task ID | Robot |
|---------|-------|
| `Unitree-G1-29dof-Velocity` | G1 (29 joints) |
| `Unitree-G1-23dof-Velocity` | G1 (23 joints) |
| `Unitree-Go2-Velocity` | Go2 |
| `Unitree-H1-Velocity` | H1 |

### Mimic (Dance)

| Task ID | Robot | Motion |
|---------|-------|--------|
| `Unitree-G1-29dof-Mimic-Gangnanm-Style` | G1-29dof | Gangnam Style |
| `Unitree-G1-29dof-Mimic-Dance-102` | G1-29dof | Dance 102 |
| `Unitree-G1-29dof-Mimic-My-Dance` | G1-29dof | Custom |
| `Unitree-G1-29dof-Mimic-Bata-Dias` | G1-29dof | Rokoko |
| `Unitree-G1-23dof-Mimic-Bata-Dias` | G1-23dof | Rokoko |

---

## Motion Capture Pipeline

Create custom dance motions from Rokoko or video:

| Guide | Description |
|-------|-------------|
| [ROKOKO_PIPELINE.md](./ROKOKO_PIPELINE.md) | Rokoko FBX → CSV → Training |
| [MOTION_CAPTURE_PIPELINE.md](./MOTION_CAPTURE_PIPELINE.md) | Video → SMPL → Robot |
| [G1_23DOF_GUIDE.md](./G1_23DOF_GUIDE.md) | G1-23dof specific guide |
| [4D_HUMANS_GUIDE.md](./4D_HUMANS_GUIDE.md) | 4D-Humans pose extraction |

### Pipeline

```
Rokoko / Video
      ↓
   FBX / MP4
      ↓
  SMPL / BVH
      ↓
 CSV (60 Hz)
      ↓
    NPZ
      ↓
  Training
      ↓
 policy.onnx
      ↓
 Real Robot
```

---

## Project Structure

```
unitree_rl_lab/
├── deploy/robots/
│   ├── g1_29dof/
│   │   ├── build/g1_ctrl          # Compiled controller
│   │   └── config/
│   │       ├── config.yaml        # FSM + gamepad mapping
│   │       └── policy/
│   │           ├── velocity/      # Walking policy
│   │           └── mimic/         # Dance policies
│   │               ├── gangnam_style/
│   │               ├── dance_102/
│   │               ├── my_dance/
│   │               └── bata_dias/
│   └── g1_23dof/
│       └── config/policy/
├── source/.../tasks/
│   ├── locomotion/               # Walking tasks
│   └── mimic/                    # Dance tasks
├── scripts/mimic/
│   ├── csv_to_npz.py            # Convert motion data
│   └── replay_motion.py         # Preview motion
└── logs/rsl_rl/                  # Training outputs
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| GPU | RTX 2070 | RTX 3080+ |
| VRAM | 8 GB | 16 GB+ |
| RAM | 32 GB | 64 GB |

---

## Documentation

| File | Description |
|------|-------------|
| [CLAUDE.md](./CLAUDE.md) | Full technical documentation |
| [ROKOKO_PIPELINE.md](./ROKOKO_PIPELINE.md) | Rokoko motion capture |
| [MOTION_CAPTURE_PIPELINE.md](./MOTION_CAPTURE_PIPELINE.md) | Video to robot motion |
| [G1_23DOF_GUIDE.md](./G1_23DOF_GUIDE.md) | G1-23dof guide |
| [SETUP_LOG.md](./SETUP_LOG.md) | Installation log |

---

## Acknowledgements

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
- [MuJoCo](https://github.com/google-deepmind/mujoco)
- [robot_lab](https://github.com/fan-ziqi/robot_lab)
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
