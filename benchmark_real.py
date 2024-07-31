import numpy as np
import open3d as o3d
import torch
import os
from gedi import GeDi
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import json
import seaborn as sns


def compute_registration(src_pcd, tgt_pcd, model, voxel_size, patches_per_pair):

    # Color the point clouds for visualization
    src_pcd.paint_uniform_color([1, 0.706, 0])
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])

    # Estimate normals (only for visualization)
    src_pcd.estimate_normals()
    tgt_pcd.estimate_normals()

    # Randomly sample points from the point cloud
    src_points = np.asarray(src_pcd.points)
    tgt_points = np.asarray(tgt_pcd.points)

    patches_per_pair = min(patches_per_pair, src_points.shape[0], tgt_points.shape[0])
    inds0 = np.random.choice(src_points.shape[0], patches_per_pair, replace=False)
    inds1 = np.random.choice(tgt_points.shape[0], patches_per_pair, replace=False)

    pts0 = torch.tensor(src_points[inds0]).float()
    pts1 = torch.tensor(tgt_points[inds1]).float()

    # Apply voxelization to the point cloud
    src_pcd = src_pcd.voxel_down_sample(voxel_size)
    tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size)

    src_points = torch.tensor(np.asarray(src_pcd.points)).float()
    tgt_points = torch.tensor(np.asarray(tgt_pcd.points)).float()

    # Compute descriptors
    src_desc = model.compute(pts=pts0, pcd=src_points)
    tgt_desc = model.compute(pts=pts1, pcd=tgt_points)

    # Prepare format for Open3D RANSAC
    src_desc_o3d = o3d.pipelines.registration.Feature()
    tgt_desc_o3d = o3d.pipelines.registration.Feature()

    src_desc_o3d.data = src_desc.T
    tgt_desc_o3d.data = tgt_desc.T

    src_pcd_ransac = o3d.geometry.PointCloud()
    src_pcd_ransac.points = o3d.utility.Vector3dVector(pts0.numpy())
    tgt_pcd_ransac = o3d.geometry.PointCloud()
    tgt_pcd_ransac.points = o3d.utility.Vector3dVector(pts1.numpy())

    # Apply RANSAC
    est_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd_ransac,
        tgt_pcd_ransac,
        src_desc_o3d,
        tgt_desc_o3d,
        mutual_filter=True,
        max_correspondence_distance=0.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.02),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
    )

    # save the source and target point clouds with the appllied transformation
    src_pcd_tranformed = src_pcd_ransac.transform(est_result.transformation)
    # color the source and the target
    src_pcd_tranformed.paint_uniform_color([1, 0, 0])
    tgt_pcd.paint_uniform_color([0, 0, 1])
    # comnine the point clouds
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(
        np.vstack((np.asarray(src_pcd_tranformed.points), np.asarray(tgt_pcd.points)))
    )
    # save the combined point cloud
    o3d.io.write_point_cloud(
        "results/real_data_benchmark/combined_pcd.ply", combined_pcd
    )

    return est_result


def get_global_min_max(directory):
    global_min = np.inf
    global_max = -np.inf

    for filename in os.listdir(directory):
        if filename.endswith(".ply"):
            pcd = o3d.io.read_point_cloud(os.path.join(directory, filename))
            points = np.asarray(pcd.points)
            min_vals = points.min(axis=0)
            max_vals = points.max(axis=0)
            global_min = min(global_min, min_vals.min())
            global_max = max(global_max, max_vals.max())

    return global_min, global_max


def normalize_point_cloud_with_global(pcd, global_min, global_max):
    points = np.asarray(pcd.points)
    normalized_points = (points - global_min) / (global_max - global_min)
    pcd.points = o3d.utility.Vector3dVector(normalized_points)
    return pcd


def load_model():
    # Configuration for GeDi
    config = {
        "dim": 32,  # Descriptor output dimension
        "samples_per_batch": 500,  # Batches to process the data on GPU
        "samples_per_patch_lrf": 4000,  # Num. of points to process with LRF
        "samples_per_patch_out": 512,  # Num. of points to sample for PointNet++
        "r_lrf": 0.5,  # LRF radius
        "fchkpt_gedi_net": "data/chkpts/3dmatch/chkpt.tar",  # Path to checkpoint
    }

    # Initialize GeDi
    gedi = GeDi(config=config)

    return gedi


def data_augmentation(
    pcd_path, rotation_range, translation_range, noise_std, num_outliers
):
    pcd = o3d.io.read_point_cloud(pcd_path)

    # Randomly generate a rotation matrix
    angle = np.random.uniform(*rotation_range)
    axis = np.array([0, 0, 1])  # Rotation around the z-axis
    rotation_matrix = pcd.get_rotation_matrix_from_axis_angle(angle * axis)

    # Randomly generate a translation vector
    translation = np.random.uniform(*translation_range, 3)

    # Combine rotation and translation into a single transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = translation

    # Apply the transformation matrix to the point cloud
    pcd.transform(transformation)

    # Add random noise to the point cloud
    noise = noise_std * np.random.randn(np.asarray(pcd.points).shape[0], 3)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + noise)

    # Add random outliers to the point cloud
    outliers = np.random.uniform(-1, 1, (num_outliers, 3))
    pcd.points = o3d.utility.Vector3dVector(
        np.vstack((np.asarray(pcd.points), outliers))
    )

    return pcd


def normalize_point_cloud(pcd):
    """
    Normalize a point cloud to fit within a unit cube centered at the origin.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        o3d.geometry.PointCloud: Normalized point cloud.
        float: Scale factor used for normalization.
    """
    # Compute the bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    scale = max(bbox.get_extent())

    # Normalize the points
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) / scale)
    return pcd, scale


def benchmark_real(input_dir, target_path, results_path, global_min, global_max):
    # Load the GeDi model
    model = load_model()

    results = []
    target_pcd = o3d.io.read_point_cloud(target_path)
    target_pcd_normalised = normalize_point_cloud_with_global(
        target_pcd, global_min, global_max
    )

    source_files = os.listdir(input_dir)
    for source in tqdm(source_files, desc="Processing point clouds", unit="file"):
        source_path = os.path.join(input_dir, source)
        source_pcd = o3d.io.read_point_cloud(source_path)
        source_pcd_normalized = normalize_point_cloud_with_global(
            source_pcd, global_min, global_max
        )

        # Compute registration result
        result = compute_registration(
            source_pcd_normalized,
            target_pcd_normalised,
            model,
            voxel_size=0.01,
            patches_per_pair=4000,
        )

        # Create a dictionary for the result
        result_dict = {
            "source_path": source_path,
            "target_path": target_path,
            "fitness": result.fitness,
            "rmse": result.inlier_rmse,
            "transformation": result.transformation.tolist(),
        }

        # Add the result to the results list
        results.append(result_dict)

    # Save results to the specified path
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)


################### data analysis ####################


def load_results(json_path):
    with open(json_path, "r") as file:
        results = json.load(file)
    return results


def prepare_data(results):
    # Extract piece number from the source_path
    df = pd.DataFrame(results)
    # Extract fragment names from the 'source_path' column
    df["fragment_name"] = df["source_path"].apply(
        lambda x: x.split("/")[-1].replace(".ply", "")
    )
    # Extract piece names from the 'target_path' column
    df["piece_name"] = df["target_path"].apply(
        lambda x: x.split("/")[-1].replace(".ply", "")
    )
    return df


def plot_results(df, target_name):

    # Determine the fragment with the lowest RMSE and the correct piece
    min_rmse_row = df[df["rmse"] == df["rmse"].min()]
    min_rmse_fragment = min_rmse_row["fragment_name"].values[0]
    min_rmse_piece = min_rmse_row["piece_name"].values[0]

    # Define color mapping based on the correctness of the lowest RMSE match
    df["rmse_color"] = "gray"  # Default color
    is_rmse_correct = (
        min_rmse_piece
        == df.loc[df["fragment_name"] == min_rmse_fragment, "piece_name"].values[0]
    )
    df.loc[df["fragment_name"] == min_rmse_fragment, "rmse_color"] = (
        "green" if is_rmse_correct else "red"
    )

    # Determine the fragment with the highest fitness and the correct piece
    max_fitness_row = df[df["fitness"] == df["fitness"].max()]
    max_fitness_fragment = max_fitness_row["fragment_name"].values[0]
    max_fitness_piece = max_fitness_row["piece_name"].values[0]

    # Define color mapping based on the correctness of the highest fitness match
    df["fitness_color"] = "gray"  # Default color
    is_fitness_correct = (
        max_fitness_piece
        == df.loc[df["fragment_name"] == max_fitness_fragment, "piece_name"].values[0]
    )
    df.loc[df["fragment_name"] == max_fitness_fragment, "fitness_color"] = (
        "green" if is_fitness_correct else "red"
    )

    # Create a single figure with two subplots
    plt.figure(figsize=(14, 12))

    # Plot fitness values
    plt.subplot(2, 1, 1)
    sns.barplot(
        x="fragment_name", y="fitness", data=df, palette=df["fitness_color"].tolist()
    )
    plt.title(
        f'Fitness Comparison Across Fragments (Target: {df["piece_name"].iloc[0]})'
    )
    plt.xlabel("Fragment Name")
    plt.ylabel("Fitness")
    plt.xticks(rotation=45)

    # Plot RMSE values with color mapping
    plt.subplot(2, 1, 2)
    sns.barplot(x="fragment_name", y="rmse", data=df, palette=df["rmse_color"].tolist())
    plt.title(f'RMSE Comparison Across Fragments (Target: {df["piece_name"].iloc[0]})')
    plt.xlabel("Fragment Name")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"results/real_data_benchmark/{target_name}/comparison.png")
    plt.show()


if __name__ == "__main__":
    source_directory = "data/real_world/Scan"
    target_directory = "data/real_world/CAD"

    # get global min and max values for normalization
    global_min_src, global_max_src = get_global_min_max(source_directory)
    global_min_tgt, global_max_tgt = get_global_min_max(target_directory)

    global_min = min(global_min_src, global_min_tgt)
    global_max = max(global_max_src, global_max_tgt)

    # List all PLY files in the input directory to use as targets
    target_files = [f for f in os.listdir(target_directory) if f.endswith(".ply")]

    # Iterate over each target file
    for target_file in target_files:
        target_path = os.path.join(target_directory, target_file)
        target_name = target_file.replace(".ply", "")
        results_json_path = (
            f"results/real_data_benchmark/{target_name}/registration_results.json"
        )
        results_pngs_path = f"results/real_data_benchmark/{target_name}"

        # Run benchmark
        benchmark_real(
            input_dir=source_directory,
            target_path=target_path,
            results_path=results_json_path,
            global_min=global_min,
            global_max=global_max,
        )

        # Load and prepare data
        results = load_results(results_json_path)
        df = prepare_data(results)

        # Now print the DataFrame
        print(df)

        # Plot the comparisons
        plot_results(df, target_name)
