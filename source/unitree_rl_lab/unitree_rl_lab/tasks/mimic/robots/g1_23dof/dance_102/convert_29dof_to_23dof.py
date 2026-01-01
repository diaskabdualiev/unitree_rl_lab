"""
Convert G1 29dof motion CSV to G1 23dof format.

G1 29dof has 36 columns: 7 (pose) + 29 (joints)
G1 23dof has 30 columns: 7 (pose) + 23 (joints)

Removed joints (6 total):
- waist_roll_joint (index 20)
- waist_pitch_joint (index 21)
- left_wrist_pitch_joint (index 27)
- left_wrist_yaw_joint (index 28)
- right_wrist_pitch_joint (index 34)
- right_wrist_yaw_joint (index 35)
"""

import numpy as np
import os

# Columns to remove (0-indexed)
COLUMNS_TO_REMOVE = [20, 21, 27, 28, 34, 35]

def convert_csv(input_path: str, output_path: str):
    """Convert 29dof CSV to 23dof CSV."""

    # Load CSV
    data = np.loadtxt(input_path, delimiter=',')
    print(f"Input shape: {data.shape} (expected 36 columns)")

    if data.shape[1] != 36:
        raise ValueError(f"Expected 36 columns, got {data.shape[1]}")

    # Remove columns
    columns_to_keep = [i for i in range(36) if i not in COLUMNS_TO_REMOVE]
    data_23dof = data[:, columns_to_keep]

    print(f"Output shape: {data_23dof.shape} (expected 30 columns)")

    # Save
    np.savetxt(output_path, data_23dof, delimiter=',', fmt='%.6f')
    print(f"Saved to: {output_path}")

    return data_23dof


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Input: 29dof dance_102
    input_csv = os.path.join(
        script_dir,
        "../../g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv"
    )

    # Output: 23dof dance_102
    output_csv = os.path.join(script_dir, "G1_Take_102_23dof_60hz.csv")

    convert_csv(input_csv, output_csv)
