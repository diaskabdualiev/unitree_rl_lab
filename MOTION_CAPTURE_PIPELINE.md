# Motion Capture Pipeline for Humanoid Robot Training

This guide explains how to create CSV motion files for training humanoid robots (G1, H1) using various motion capture and retargeting methods.

## Overview Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MOTION SOURCES                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│   │   VIDEO      │    │   MOTION     │    │   DATASETS   │             │
│   │  (YouTube,   │    │   CAPTURE    │    │  (LAFAN1,    │             │
│   │   camera)    │    │   (suit)     │    │   AMASS)     │             │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘             │
│          │                   │                   │                     │
│          ▼                   ▼                   │                     │
│   ┌──────────────┐    ┌──────────────┐           │                     │
│   │ WHAM / 4D    │    │   BVH file   │           │                     │
│   │ Humans /     │    │              │           │                     │
│   │ GVHMR        │    │              │           │                     │
│   └──────┬───────┘    └──────┬───────┘           │                     │
│          │                   │                   │                     │
│          ▼                   ▼                   ▼                     │
│   ┌──────────────────────────────────────────────────┐                 │
│   │              SMPL / SMPLX format                 │                 │
│   │       (universal human representation)           │                 │
│   └──────────────────────┬───────────────────────────┘                 │
│                          │                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RETARGETING                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────────────────────────────────────┐                 │
│   │         GMR (General Motion Retargeting)         │                 │
│   │    github.com/YanjieZe/GMR                       │                 │
│   │                                                  │                 │
│   │  Supports: Unitree G1, H1, H1_2 + 12 robots     │                 │
│   │  Input: SMPLX, BVH (LAFAN1, Nokov), FBX         │                 │
│   │  Output: CSV with joint positions               │                 │
│   └──────────────────────┬───────────────────────────┘                 │
│                          │                                              │
│   OR                     │                                              │
│                          │                                              │
│   ┌──────────────────────────────────────────────────┐                 │
│   │    Humanoid-Motion-Retargeting (Unitree)         │                 │
│   │    github.com/XinLang2019/Humanoid-Motion-Retarg │                 │
│   │                                                  │                 │
│   │  Method: Interaction Mesh + IK optimization     │                 │
│   │  Input: LAFAN1 motion capture                   │                 │
│   │  Output: CSV (30 FPS)                           │                 │
│   └──────────────────────┬───────────────────────────┘                 │
│                          │                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CSV → NPZ → TRAINING                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CSV file                 NPZ file                 Policy              │
│   (x,y,z,quat,joints) --→ (+ body_pos_w) -------→ (policy.onnx)        │
│                                                                         │
│   csv_to_npz.py            train.py                                    │
│   (Isaac Lab simulation)   (RL training)                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Method Comparison

| Method | Source | Quality | Complexity | G1 Support |
|--------|--------|---------|------------|------------|
| **WHAM + GMR** | Any video | Medium | Medium | Yes |
| **Mixamo + GMR** | Ready animations | High | Low | Yes |
| **LAFAN1 + GMR** | MoCap dataset | Very High | Low | Yes |
| **PHC** | Video | High (physics) | High | Yes |
| **XR Teleoperation** | VR controller | High | Medium | Arms only |

---

## Method 1: Video → SMPL → Robot (Most Flexible)

Extract motion from any video and retarget to robot.

### Step 1: Extract Motion from Video using WHAM

[WHAM](https://github.com/yohanshin/WHAM) (CVPR 2024) reconstructs world-grounded humans with accurate 3D motion.

```bash
# Install WHAM
git clone https://github.com/yohanshin/WHAM
cd WHAM
conda create -n wham python=3.10 -y
conda activate wham
pip install -r requirements.txt

# Download pretrained models
bash fetch_demo_data.sh

# Extract motion from video
python demo.py \
  --video /path/to/dance_video.mp4 \
  --output_pth output/ \
  --save_pkl
```

**Output:** SMPL parameters (pose, shape, translation) in PKL format

### Step 2: Retarget to G1 Robot using GMR

[GMR](https://github.com/YanjieZe/GMR) performs real-time high-quality retargeting.

```bash
# Install GMR
git clone https://github.com/YanjieZe/GMR
cd GMR
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .

# Download SMPL-X body models
# Place in assets/body_models/smplx/
# Get from: https://smpl-x.is.tue.mpg.de/

# Retarget SMPL → G1
python scripts/smpl_to_robot.py \
  --input output/smpl_motion.pkl \
  --robot unitree_g1 \
  --output g1_motion.csv
```

### Step 3: Convert to NPZ and Train

```bash
cd /path/to/unitree_rl_lab
conda activate env_isaaclab

# Convert CSV to NPZ (for 29dof)
python scripts/mimic/csv_to_npz.py \
  -f g1_motion.csv \
  --input_fps 30 \
  --robot g1_29dof

# Or for 23dof (after converting CSV)
python scripts/mimic/csv_to_npz.py \
  -f g1_motion_23dof.csv \
  --input_fps 30 \
  --robot g1_23dof

# Train
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Mimic-Custom \
  --headless
```

---

## Method 2: Mixamo/BVH → Robot (Easiest)

Use pre-made animations from Mixamo.

### Step 1: Download Animation from Mixamo

1. Go to [mixamo.com](https://www.mixamo.com) (free account required)
2. Choose any character
3. Browse animations (dance, walk, gestures, etc.)
4. Download as **FBX** format
5. Convert to BVH using Blender (optional)

### Step 2: Retarget to G1 using GMR

```bash
conda activate gmr
cd GMR

# GMR supports BVH directly
python scripts/bvh_to_robot.py \
  --input dance.bvh \
  --format lafan1 \
  --robot unitree_g1 \
  --output g1_dance.csv

# Visualize in MuJoCo (optional)
python scripts/bvh_to_robot.py \
  --input dance.bvh \
  --robot unitree_g1 \
  --visualize
```

### Supported BVH Formats

| Format | Source | Notes |
|--------|--------|-------|
| `lafan1` | LAFAN1 dataset | Default format |
| `nokov` | Nokov MoCap system | Chinese MoCap |
| `mixamo` | Mixamo.com | May need conversion |

---

## Method 3: LAFAN1 Dataset (Highest Quality)

LAFAN1 is the dataset Unitree officially uses for their demos.

### Download LAFAN1

```bash
# Clone the dataset
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset
cd ubisoft-laforge-animation-dataset

# Dataset structure:
# lafan1/
# ├── aiming1_subject1.bvh
# ├── dance1_subject1.bvh
# ├── ...
```

### Batch Retarget to G1

```bash
conda activate gmr
cd GMR

# Retarget all LAFAN1 motions
python scripts/batch_retarget.py \
  --input_dir /path/to/lafan1/ \
  --robot unitree_g1 \
  --output_dir g1_motions/

# This creates:
# g1_motions/
# ├── aiming1_subject1.csv
# ├── dance1_subject1.csv
# ├── ...
```

### Available Motion Types in LAFAN1

| Category | Examples |
|----------|----------|
| **Dance** | dance1, dance2, dance3 |
| **Walk** | walk1, walk2, walk3 |
| **Run** | run1, run2 |
| **Jump** | jump1, jump2 |
| **Fight** | fight1, fight2 |
| **Ground** | ground1, ground2 |

---

## Method 4: PHC/PULSE (Physics-Based from Video)

[PHC](https://github.com/ZhengyiLuo/PHC) extracts motion from video with physics simulation.

### Features

- Handles noisy video input
- Physics-based motion correction
- Supports Unitree G1 and H1 (added October 2024)
- Retargeting documentation (added December 2024)

### Installation

```bash
git clone https://github.com/ZhengyiLuo/PHC
cd PHC
conda create -n phc python=3.8 -y
conda activate phc
pip install -r requirements.txt

# Install Isaac Gym
# Download from NVIDIA and install
pip install isaacgym
```

### Extract and Retarget

```bash
# Process video
python scripts/demo_video.py \
  --video_path /path/to/video.mp4 \
  --output_dir output/

# Retarget to G1 (documentation added December 2024)
python scripts/retarget_to_robot.py \
  --input output/motion.pkl \
  --robot unitree_g1 \
  --output g1_motion.csv
```

---

## Method 5: AMASS Dataset (Largest Collection)

[AMASS](https://amass.is.tue.mpg.de/) unifies 15+ motion capture datasets.

### Download AMASS

1. Register at https://amass.is.tue.mpg.de/
2. Download desired subsets:
   - CMU (largest)
   - BMLrub
   - KIT
   - Eyes Japan
   - etc.

### Convert AMASS to Robot

```bash
conda activate gmr
cd GMR

# AMASS uses SMPLX format (directly supported)
python scripts/amass_to_robot.py \
  --input /path/to/amass/CMU/01/01_01_stageii.npz \
  --robot unitree_g1 \
  --output g1_motion.csv
```

---

## Method 6: Humanoid-Motion-Retargeting (Unitree Official)

[Official Unitree tool](https://github.com/XinLang2019/Humanoid-Motion-Retargeting) for retargeting.

### Features

- Uses Interaction Mesh + IK optimization
- Considers joint position and velocity constraints
- Officially supports G1, H1, H1_2

### Installation

```bash
git clone https://github.com/XinLang2019/Humanoid-Motion-Retargeting
cd Humanoid-Motion-Retargeting
conda create -n retarget python=3.8 -y
conda activate retarget
pip install -r requirements.txt

# Install Isaac Gym Preview 3
```

### Usage

```bash
# Retarget LAFAN1 to G1
python retarget.py \
  --robot g1 \
  --input_dir /path/to/lafan1/ \
  --output_dir output/

# Convert CSV to PKL for training
python csv_to_pkl.py \
  --input output/dance1.csv \
  --output output/dance1.pkl
```

### CSV Output Format

```
# Each row = one frame at 30 FPS
# Columns:
# 0-2: root_x, root_y, root_z
# 3-6: root_qx, root_qy, root_qz, root_qw
# 7+: joint angles (29 for G1-29dof, 23 for G1-23dof)
```

---

## XR Teleoperation (Arms Only)

XR Teleoperation records arm movements but NOT full body.

### What It Records

```
XR Teleoperate captures:
├── Arm joint positions (left/right)
├── Hand/gripper positions
└── Does NOT capture legs (robot stands still)
```

### When to Use

| Use Case | XR Teleoperation | MoCap/Video |
|----------|------------------|-------------|
| Dancing | No | Yes |
| Walking | No | Yes |
| Pick & Place | Yes | No |
| Manipulation | Yes | No |
| Gestures (arms) | Yes | Overkill |

### Hybrid Approach

Combine XR teleoperation for arms with generated leg motion:

```python
# Pseudocode
arm_motion = load_xr_recording("arms.csv")
leg_motion = generate_standing_motion(duration=arm_motion.duration)
full_motion = merge(leg_motion, arm_motion)
save_csv(full_motion, "full_body.csv")
```

---

## CSV File Format Reference

### G1-29dof CSV (36 columns)

```csv
# x, y, z, qx, qy, qz, qw, j1, j2, ..., j29
0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, -0.1, 0.0, ...
```

| Columns | Content |
|---------|---------|
| 0-2 | Root position (x, y, z) |
| 3-6 | Root quaternion (qx, qy, qz, qw) |
| 7-35 | Joint positions (29 joints) |

### G1-23dof CSV (30 columns)

```csv
# x, y, z, qx, qy, qz, qw, j1, j2, ..., j23
0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, -0.1, 0.0, ...
```

| Columns | Content |
|---------|---------|
| 0-2 | Root position (x, y, z) |
| 3-6 | Root quaternion (qx, qy, qz, qw) |
| 7-29 | Joint positions (23 joints) |

### Converting 29dof → 23dof

Remove columns (0-indexed from joint start):
- 13: waist_roll_joint
- 14: waist_pitch_joint
- 20: left_wrist_pitch_joint
- 21: left_wrist_yaw_joint
- 27: right_wrist_pitch_joint
- 28: right_wrist_yaw_joint

---

## Recommended Workflows

### Quick Start (Easiest)
```
Mixamo → Download FBX → Blender → BVH → GMR → CSV → csv_to_npz.py → Train
```

### Custom Dance from Video
```
Record video → WHAM → SMPL → GMR → CSV → csv_to_npz.py → Train
```

### Maximum Quality
```
LAFAN1 dataset → GMR → CSV → csv_to_npz.py → Train
```

### Physics-Corrected Motion
```
Video → PHC → Physics-corrected SMPL → GMR → CSV → csv_to_npz.py → Train
```

---

## Troubleshooting

### BVH Import Errors

**Problem:** GMR doesn't recognize BVH format
```
Error: Unknown BVH format
```

**Solution:** Specify format explicitly or convert in Blender:
```bash
python scripts/bvh_to_robot.py --input motion.bvh --format lafan1
```

### Joint Count Mismatch

**Problem:** CSV has wrong number of joints
```
ValueError: Expected 29 joints, got 23
```

**Solution:** Use correct robot config:
```bash
# For 29dof robot
python csv_to_npz.py -f motion.csv --robot g1_29dof

# For 23dof robot (after converting CSV)
python csv_to_npz.py -f motion_23dof.csv --robot g1_23dof
```

### SMPL-X Model Missing

**Problem:** GMR can't find body models
```
FileNotFoundError: assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

**Solution:** Download from https://smpl-x.is.tue.mpg.de/ and place in `assets/body_models/smplx/`

### Motion Too Fast/Slow

**Problem:** Robot moves unrealistically fast or slow

**Solution:** Adjust FPS during conversion:
```bash
# Original video was 24 FPS, target is 50 FPS
python csv_to_npz.py -f motion.csv --input_fps 24 --output_fps 50
```

---

## Resources

### Tools

| Tool | URL | Purpose |
|------|-----|---------|
| GMR | https://github.com/YanjieZe/GMR | General motion retargeting |
| WHAM | https://github.com/yohanshin/WHAM | Video → SMPL |
| PHC | https://github.com/ZhengyiLuo/PHC | Physics-based control |
| Humanoid-Motion-Retargeting | https://github.com/XinLang2019/Humanoid-Motion-Retargeting | Unitree official |

### Datasets

| Dataset | URL | Content |
|---------|-----|---------|
| LAFAN1 | https://github.com/ubisoft/ubisoft-laforge-animation-dataset | High-quality MoCap |
| AMASS | https://amass.is.tue.mpg.de/ | Unified MoCap collection |
| Mixamo | https://www.mixamo.com | Free animations |
| CMU MoCap | http://mocap.cs.cmu.edu/ | Classic MoCap database |

### Papers

- [WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion](https://arxiv.org/abs/2312.07531) (CVPR 2024)
- [PHC: Perpetual Humanoid Control for Real-time Simulated Avatars](https://github.com/ZhengyiLuo/PHC) (ICCV 2023)
- [PULSE: Universal Humanoid Motion Representations](https://github.com/ZhengyiLuo/PULSE) (ICLR 2024)
- [GMR: General Motion Retargeting](https://arxiv.org/abs/2510.02252) (arXiv 2025)
