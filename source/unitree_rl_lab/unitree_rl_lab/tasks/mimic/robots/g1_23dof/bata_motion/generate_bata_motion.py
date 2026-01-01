"""
Generate bata_motion CSV for G1-23dof robot.

Motion description:
An elderly person sitting calmly. Slowly raises both hands in front of chest,
palms facing upward, fingers relaxed. Holds this position with gentle subtle
swaying. Then brings both palms to face, touching forehead, and slowly draws
hands down across the face to chin in a single smooth motion. Lowers hands
to rest on knees. Movements are slow, deliberate, ceremonial.

Since G1 is a standing humanoid, we interpret "sitting" as a calm standing pose
with slightly bent knees, and "rest on knees" as hands lowered to sides.
"""

import numpy as np
import os

# G1-23dof joint names (in order)
JOINT_NAMES = [
    "left_hip_pitch_joint",      # 0
    "left_hip_roll_joint",       # 1
    "left_hip_yaw_joint",        # 2
    "left_knee_joint",           # 3
    "left_ankle_pitch_joint",    # 4
    "left_ankle_roll_joint",     # 5
    "right_hip_pitch_joint",     # 6
    "right_hip_roll_joint",      # 7
    "right_hip_yaw_joint",       # 8
    "right_knee_joint",          # 9
    "right_ankle_pitch_joint",   # 10
    "right_ankle_roll_joint",    # 11
    "waist_yaw_joint",           # 12
    "left_shoulder_pitch_joint", # 13
    "left_shoulder_roll_joint",  # 14
    "left_shoulder_yaw_joint",   # 15
    "left_elbow_joint",          # 16
    "left_wrist_roll_joint",     # 17
    "right_shoulder_pitch_joint",# 18
    "right_shoulder_roll_joint", # 19
    "right_shoulder_yaw_joint",  # 20
    "right_elbow_joint",         # 21
    "right_wrist_roll_joint",    # 22
]

FPS = 60
N_JOINTS = 23


def smooth_interpolate(start, end, t):
    """Smooth interpolation using cosine easing."""
    # Cosine easing for smooth start and end
    smooth_t = (1 - np.cos(t * np.pi)) / 2
    return start + (end - start) * smooth_t


def generate_phase(start_pose, end_pose, duration_sec, fps):
    """Generate frames for a phase with smooth interpolation."""
    n_frames = int(duration_sec * fps)
    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 0
        pose = smooth_interpolate(start_pose, end_pose, t)
        frames.append(pose)
    return frames


def add_subtle_sway(frames, amplitude=0.02, frequency=0.5, fps=60):
    """Add subtle swaying motion to frames."""
    swayed_frames = []
    for i, frame in enumerate(frames):
        t = i / fps
        sway = amplitude * np.sin(2 * np.pi * frequency * t)

        new_frame = frame.copy()
        # Sway waist slightly
        new_frame[12] += sway * 0.3  # waist_yaw
        # Sway shoulders slightly (in phase)
        new_frame[13] += sway * 0.1  # left_shoulder_pitch
        new_frame[18] += sway * 0.1  # right_shoulder_pitch

        swayed_frames.append(new_frame)
    return swayed_frames


def generate_bata_motion():
    """Generate the complete bata_motion sequence."""

    # =========================================================================
    # Define key poses (joint positions in radians)
    # =========================================================================

    # Base standing pose (calm, slightly bent knees)
    base_pose = np.zeros(N_JOINTS)
    base_pose[0] = -0.15   # left_hip_pitch (slight bend)
    base_pose[3] = 0.3     # left_knee (bent)
    base_pose[4] = -0.15   # left_ankle_pitch
    base_pose[6] = -0.15   # right_hip_pitch
    base_pose[9] = 0.3     # right_knee
    base_pose[10] = -0.15  # right_ankle_pitch
    # Arms relaxed at sides
    base_pose[13] = 0.2    # left_shoulder_pitch (slightly forward)
    base_pose[14] = 0.1    # left_shoulder_roll (slightly out)
    base_pose[16] = 0.3    # left_elbow (slightly bent)
    base_pose[18] = 0.2    # right_shoulder_pitch
    base_pose[19] = -0.1   # right_shoulder_roll
    base_pose[21] = 0.3    # right_elbow

    # Pose 1: Hands raised to chest, palms up
    chest_pose = base_pose.copy()
    chest_pose[13] = 0.8   # left_shoulder_pitch (raised forward)
    chest_pose[14] = 0.3   # left_shoulder_roll (slightly out)
    chest_pose[15] = 0.2   # left_shoulder_yaw (rotate inward)
    chest_pose[16] = 1.8   # left_elbow (bent, hands toward chest)
    chest_pose[17] = 1.57  # left_wrist_roll (palm up = 90 degrees)
    chest_pose[18] = 0.8   # right_shoulder_pitch
    chest_pose[19] = -0.3  # right_shoulder_roll
    chest_pose[20] = -0.2  # right_shoulder_yaw
    chest_pose[21] = 1.8   # right_elbow
    chest_pose[22] = -1.57 # right_wrist_roll (palm up)

    # Pose 2: Hands at forehead (prayer-like)
    forehead_pose = base_pose.copy()
    forehead_pose[13] = 1.2   # left_shoulder_pitch (raised higher)
    forehead_pose[14] = 0.5   # left_shoulder_roll
    forehead_pose[15] = 0.4   # left_shoulder_yaw (hands toward face)
    forehead_pose[16] = 2.2   # left_elbow (more bent)
    forehead_pose[17] = 0.0   # left_wrist_roll (palms facing face)
    forehead_pose[18] = 1.2   # right_shoulder_pitch
    forehead_pose[19] = -0.5  # right_shoulder_roll
    forehead_pose[20] = -0.4  # right_shoulder_yaw
    forehead_pose[21] = 2.2   # right_elbow
    forehead_pose[22] = 0.0   # right_wrist_roll

    # Pose 3: Hands at chin (after drawing down face)
    chin_pose = base_pose.copy()
    chin_pose[13] = 0.6    # left_shoulder_pitch (lowered)
    chin_pose[14] = 0.4    # left_shoulder_roll
    chin_pose[15] = 0.3    # left_shoulder_yaw
    chin_pose[16] = 2.0    # left_elbow
    chin_pose[17] = 0.0    # left_wrist_roll
    chin_pose[18] = 0.6    # right_shoulder_pitch
    chin_pose[19] = -0.4   # right_shoulder_roll
    chin_pose[20] = -0.3   # right_shoulder_yaw
    chin_pose[21] = 2.0    # right_elbow
    chin_pose[22] = 0.0    # right_wrist_roll

    # Pose 4: Hands resting low (on "knees" / at sides)
    rest_pose = base_pose.copy()
    rest_pose[13] = 0.4    # left_shoulder_pitch
    rest_pose[14] = 0.05   # left_shoulder_roll (close to body)
    rest_pose[15] = 0.0    # left_shoulder_yaw
    rest_pose[16] = 0.5    # left_elbow (relaxed)
    rest_pose[17] = 0.0    # left_wrist_roll
    rest_pose[18] = 0.4    # right_shoulder_pitch
    rest_pose[19] = -0.05  # right_shoulder_roll
    rest_pose[20] = 0.0    # right_shoulder_yaw
    rest_pose[21] = 0.5    # right_elbow
    rest_pose[22] = 0.0    # right_wrist_roll

    # =========================================================================
    # Generate motion phases
    # =========================================================================

    all_frames = []

    # Phase 0: Initial calm pose (2 seconds)
    phase0 = generate_phase(base_pose, base_pose, 2.0, FPS)
    phase0 = add_subtle_sway(phase0, amplitude=0.01, frequency=0.3)
    all_frames.extend(phase0)

    # Phase 1: Slowly raise hands to chest (4 seconds)
    phase1 = generate_phase(base_pose, chest_pose, 4.0, FPS)
    all_frames.extend(phase1)

    # Phase 2: Hold at chest with gentle swaying (4 seconds)
    phase2 = generate_phase(chest_pose, chest_pose, 4.0, FPS)
    phase2 = add_subtle_sway(phase2, amplitude=0.03, frequency=0.4)
    all_frames.extend(phase2)

    # Phase 3: Bring hands to forehead (3 seconds)
    phase3 = generate_phase(chest_pose, forehead_pose, 3.0, FPS)
    all_frames.extend(phase3)

    # Phase 4: Hold at forehead briefly (1 second)
    phase4 = generate_phase(forehead_pose, forehead_pose, 1.0, FPS)
    all_frames.extend(phase4)

    # Phase 5: Draw hands down face to chin (3 seconds)
    phase5 = generate_phase(forehead_pose, chin_pose, 3.0, FPS)
    all_frames.extend(phase5)

    # Phase 6: Lower hands to rest (3 seconds)
    phase6 = generate_phase(chin_pose, rest_pose, 3.0, FPS)
    all_frames.extend(phase6)

    # Phase 7: Hold final rest position (2 seconds)
    phase7 = generate_phase(rest_pose, rest_pose, 2.0, FPS)
    phase7 = add_subtle_sway(phase7, amplitude=0.01, frequency=0.25)
    all_frames.extend(phase7)

    return all_frames


def create_csv(frames, output_path):
    """Create CSV file with root pose and joint positions."""

    n_frames = len(frames)

    # Root position (standing in place, slight height for bent knees)
    root_x = np.zeros(n_frames)
    root_y = np.zeros(n_frames)
    root_z = np.full(n_frames, 0.75)  # Standing height (slightly lower for bent knees)

    # Root orientation (facing forward, quaternion xyzw)
    root_qx = np.zeros(n_frames)
    root_qy = np.zeros(n_frames)
    root_qz = np.zeros(n_frames)
    root_qw = np.ones(n_frames)

    # Build data array
    # Columns: x, y, z, qx, qy, qz, qw, joint1, joint2, ..., joint23
    data = np.zeros((n_frames, 7 + N_JOINTS))
    data[:, 0] = root_x
    data[:, 1] = root_y
    data[:, 2] = root_z
    data[:, 3] = root_qx
    data[:, 4] = root_qy
    data[:, 5] = root_qz
    data[:, 6] = root_qw

    for i, frame in enumerate(frames):
        data[i, 7:] = frame

    # Save CSV
    np.savetxt(output_path, data, delimiter=',', fmt='%.6f')

    print(f"Created: {output_path}")
    print(f"  Frames: {n_frames}")
    print(f"  Duration: {n_frames / FPS:.2f} seconds")
    print(f"  FPS: {FPS}")
    print(f"  Columns: {data.shape[1]} (7 root + {N_JOINTS} joints)")


def main():
    # Generate motion
    print("Generating bata_motion for G1-23dof...")
    print()
    print("Motion description:")
    print("  - Phase 0: Initial calm pose (2s)")
    print("  - Phase 1: Raise hands to chest, palms up (4s)")
    print("  - Phase 2: Hold with gentle swaying (4s)")
    print("  - Phase 3: Bring hands to forehead (3s)")
    print("  - Phase 4: Hold at forehead (1s)")
    print("  - Phase 5: Draw hands down face to chin (3s)")
    print("  - Phase 6: Lower hands to rest (3s)")
    print("  - Phase 7: Final rest position (2s)")
    print()

    frames = generate_bata_motion()

    # Save CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "bata_motion_23dof_60hz.csv")

    create_csv(frames, csv_path)

    print()
    print("Next steps:")
    print("  1. Convert to NPZ:")
    print(f"     python scripts/mimic/csv_to_npz.py -f {csv_path} --input_fps 60 --robot g1_23dof")
    print()
    print("  2. Train:")
    print("     python scripts/rsl_rl/train.py --task Unitree-G1-23dof-Mimic-Bata-Motion --headless")


if __name__ == "__main__":
    main()
