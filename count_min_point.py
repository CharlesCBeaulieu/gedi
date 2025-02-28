#!/usr/bin/env python3
import os
import sys
import open3d as o3d
import numpy as np


def main():
    # Check if a folder path was provided as a command-line argument.
    # Otherwise, use the default folder "./point_clouds".
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "gedi_data/gen_scan"

    # Verify that the folder exists.
    if not os.path.isdir(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        sys.exit(1)

    # List all files in the folder with .ply or .pcd extensions.
    pcd_files = [f for f in os.listdir(folder) if f.lower().endswith((".ply", ".pcd"))]

    if not pcd_files:
        print(f"No point cloud files (.ply or .pcd) found in folder '{folder}'.")
        sys.exit(0)

    point_counts = []

    # Process each file.
    for filename in pcd_files:
        file_path = os.path.join(folder, filename)
        pcd = o3d.io.read_point_cloud(file_path)
        num_points = np.asarray(pcd.points).shape[0]
        point_counts.append(num_points)
        print(f"{filename}: {num_points} points")

    # Compute minimum and mean point counts.
    min_points = np.min(point_counts)
    mean_points = np.mean(point_counts)

    print("\nSummary:")
    print(f"Minimum number of points: {min_points}")
    print(f"Mean number of points: {mean_points:.2f}")


if __name__ == "__main__":
    main()
