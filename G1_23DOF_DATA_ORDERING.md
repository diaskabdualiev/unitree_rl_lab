# G1-23DOF Data Ordering Problem & Solution

## Problem Summary

When converting G1 from 29DOF to 23DOF, CSV motion files stop working correctly because **column indices no longer match SDK motor indices**.

## Root Cause

### Two Different Joint Ordering Systems

**1. SDK Order (27 motors, physical robot):**
```
0-5:   left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
6-11:  right leg
12:    waist_yaw
13-14: waist_roll/pitch      ← NOT IN 23DOF
15-19: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll)
20-21: left wrist_pitch/yaw  ← NOT IN 23DOF
22-26: right arm
```

**2. Isaac Lab Order (23 joints, training/simulation):**
```
Index → SDK Motor
0  → 0   (left_hip_pitch)
1  → 6   (right_hip_pitch)
2  → 12  (waist_yaw)
3  → 1   (left_hip_roll)
4  → 7   (right_hip_roll)
5  → 15  (left_shoulder_pitch)
6  → 22  (right_shoulder_pitch)
7  → 2   (left_hip_yaw)
8  → 8   (right_hip_yaw)
9  → 16  (left_shoulder_roll)
10 → 23  (right_shoulder_roll)
11 → 3   (left_knee)
12 → 9   (right_knee)
13 → 17  (left_shoulder_yaw)
14 → 24  (right_shoulder_yaw)
15 → 4   (left_ankle_pitch)
16 → 10  (right_ankle_pitch)
17 → 18  (left_elbow)
18 → 25  (right_elbow)
19 → 5   (left_ankle_roll)
20 → 11  (right_ankle_roll)
21 → 19  (left_wrist_roll)
22 → 26  (right_wrist_roll)
```

### The Mapping Array (joint_ids_map)

```cpp
static const std::vector<int> JOINT_IDS_MAP = {
    0, 6, 12,       // left_hip_pitch, right_hip_pitch, waist_yaw
    1, 7,           // left_hip_roll, right_hip_roll
    15, 22,         // left_shoulder_pitch, right_shoulder_pitch
    2, 8,           // left_hip_yaw, right_hip_yaw
    16, 23,         // left_shoulder_roll, right_shoulder_roll
    3, 9,           // left_knee, right_knee
    17, 24,         // left_shoulder_yaw, right_shoulder_yaw
    4, 10,          // left_ankle_pitch, right_ankle_pitch
    18, 25,         // left_elbow, right_elbow
    5, 11,          // left_ankle_roll, right_ankle_roll
    19, 26,         // left_wrist_roll, right_wrist_roll
};
```

### What Went Wrong

Original 29DOF CSV had 36 columns (7 pose + 29 joints).

Simple removal of 6 joints (columns 20, 21, 27, 28, 34, 35) created 30 columns (7 pose + 23 joints), BUT:
- CSV columns were in "compressed SDK order" (just removed gaps)
- C++ code expected Isaac Lab order (uses `joint_ids_map` for remapping)
- Result: joints received wrong position commands

**Example of mismatch:**
```
CSV column 7 (first joint) = left_hip_pitch (SDK 0) ← CORRECT
CSV column 8 = left_hip_roll (SDK 1)
...but Isaac Lab order expects:
Column 7 = left_hip_pitch (SDK 0)
Column 8 = right_hip_pitch (SDK 6)  ← MISMATCH!
```

## Solution We Used (Simple)

**Recreate CSV files in Isaac Lab joint order.**

Created script: `scripts/mimic/convert_29dof_to_23dof_isaac_order.py`

```python
# Isaac Lab order mapping for 23DOF
# Maps Isaac Lab index → 29DOF SDK index (column offset +7 for pose)
ISAAC_TO_29DOF_SDK = [
    0, 6, 12,       # hip_pitch L/R, waist_yaw
    1, 7,           # hip_roll L/R
    15, 22,         # shoulder_pitch L/R
    2, 8,           # hip_yaw L/R
    16, 23,         # shoulder_roll L/R
    3, 9,           # knee L/R
    17, 24,         # shoulder_yaw L/R
    4, 10,          # ankle_pitch L/R
    18, 25,         # elbow L/R
    5, 11,          # ankle_roll L/R
    19, 26,         # wrist_roll L/R
]

# For each row in 29DOF CSV:
# 1. Keep pose columns 0-6 unchanged
# 2. Reorder joint columns 7+ according to Isaac Lab order
for isaac_idx, sdk_29_idx in enumerate(ISAAC_TO_29DOF_SDK):
    output_row[7 + isaac_idx] = input_row[7 + sdk_29_idx]
```

**Converted files:**
- `bata_dias_23dof_60hz.csv`
- `gangnam_style_23dof_60hz.csv`
- `dance_102_23dof_60hz.csv`

## Alternative Solution (Complex)

**Modify C++ code to accept SDK-order CSV and remap internally.**

Would require changes to:
1. `State_Replay.cpp` - add remapping when reading CSV
2. `State_Mimic.h` (MotionLoader_) - add remapping in constructor

```cpp
// In MotionLoader_ constructor, after loading CSV:
// Remap from SDK order to Isaac Lab order
Eigen::VectorXf remapped(23);
for (int isaac_idx = 0; isaac_idx < 23; ++isaac_idx) {
    int sdk_idx = JOINT_IDS_MAP[isaac_idx];
    // Find which CSV column has this SDK index
    remapped[isaac_idx] = raw_joints[sdk_to_csv_map[sdk_idx]];
}
```

**Why we didn't use this:**
- More complex, error-prone
- CSV format becomes non-standard
- Harder to debug (can't visually inspect CSV)
- Isaac Lab training already uses Isaac Lab order

## Additional Issue: kp/kd Arrays

### Problem

`config.yaml` had 23-element kp/kd arrays, but `State_FixStand` and `State_Replay` use direct SDK indexing:
```cpp
lowcmd->msg_.motor_cmd()[i].kp() = kp[i];  // i = 0..26
```

With only 23 elements, motors 22-26 (right arm) never received gains!

### Solution

Changed all kp/kd arrays to 27 elements in SDK order:

```yaml
kp: [
  100., 100., 100., 150., 40., 40.,  # SDK 0-5: left leg
  100., 100., 100., 150., 40., 40.,  # SDK 6-11: right leg
  200,                                # SDK 12: waist_yaw
  0, 0,                               # SDK 13-14: not in 23dof
  40, 40, 40, 40, 40,                 # SDK 15-19: left arm
  0, 0,                               # SDK 20-21: not in 23dof
  40, 40, 40, 40, 40,                 # SDK 22-26: right arm
]
```

## Files Modified

| File | Change |
|------|--------|
| `scripts/mimic/convert_29dof_to_23dof_isaac_order.py` | Created - CSV conversion script |
| `config/policy/mimic/*/params/*_23dof_60hz.csv` | Recreated in Isaac Lab order |
| `src/State_Replay.cpp` | Added JOINT_IDS_MAP, fixed kp/kd to 27 elements |
| `include/State_Mimic.h` | Added documentation about CSV format |
| `config/config.yaml` | Changed all kp/kd to 27 elements in SDK order |

## Key Takeaways

1. **Always check joint ordering** when converting between DOF configurations
2. **CSV files for deployment must match the order expected by C++ code**
3. **State_Mimic uses `joint_ids_map`** (Isaac Lab order) for motor commands
4. **State_FixStand/Passive use direct SDK indexing** (need 27-element arrays)
5. **State_Replay now uses JOINT_IDS_MAP** for CSV data, but direct indexing for kp/kd

## Quick Reference

```
Isaac Lab CSV format (30 columns):
[x, y, z, qx, qy, qz, qw, j0, j1, ..., j22]
                        └── 23 joints in Isaac Lab order

SDK motor array (27 elements):
[0..5: L_leg, 6..11: R_leg, 12: waist, 13-14: SKIP, 15..19: L_arm, 20-21: SKIP, 22..26: R_arm]
```
