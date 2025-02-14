#!/usr/bin/env python3
import os
import glob
import open3d as o3d


def scale_cads(input_folder, output_folder, scale_factor=0.001):
    """
    Load every CAD file (assumed to be in PLY format) from the input_folder,
    scale them to convert from millimeters to meters, and save the converted
    point clouds in the output_folder.

    Args:
        input_folder (str): Path to the folder containing the original CAD files.
        output_folder (str): Path to the folder where converted files will be saved.
        scale_factor (float): Factor by which to scale the point clouds.
                              For mm to meters, use 0.001.
    """
    os.makedirs(output_folder, exist_ok=True)
    pattern = os.path.join(input_folder, "*.ply")
    cad_files = glob.glob(pattern)

    if not cad_files:
        print("No CAD files found in", input_folder)
        return

    for cad_file in cad_files:
        print("Processing:", cad_file)
        # Read the point cloud.
        pcd = o3d.io.read_point_cloud(cad_file)
        # Use the point cloud's center as pivot for scaling.
        center = pcd.get_center()
        pcd.scale(scale_factor, center=center)
        # Build the output file path.
        filename = os.path.basename(cad_file)
        output_path = os.path.join(output_folder, filename)
        # Save the scaled point cloud.
        o3d.io.write_point_cloud(output_path, pcd)
        print("Saved converted file to:", output_path)


if __name__ == "__main__":
    input_folder = "gedi_data/gt_cad"
    output_folder = "gedi_data/gt_cad_scaled_0001"
    scale_cads(input_folder, output_folder)
