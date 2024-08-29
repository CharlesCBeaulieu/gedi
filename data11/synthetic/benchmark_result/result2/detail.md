# Détail du test

Test porté sur 10 samples d'un sous dataset des cad de sbi, chaque cad à été utilisé comme target 1 fois, puis les autre scan on été augmenté. Voici le script utilisé pour ces tests : 

```python
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

def compute_registration(src_pcd, tgt_pcd, model, voxel_size, patches_per_pair):
    # Load point clouds
    pcd0 = src_pcd
    pcd1 = tgt_pcd

    # Color the point clouds for visualization
    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])

    # Estimate normals (only for visualization)
    pcd0.estimate_normals()
    pcd1.estimate_normals()

    # Randomly sample points from the point cloud
    inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=False)
    inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches_per_pair, replace=False)

    pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
    pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

    # Apply voxelization to the point cloud
    pcd0 = pcd0.voxel_down_sample(voxel_size)
    pcd1 = pcd1.voxel_down_sample(voxel_size)

    _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
    _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

    # Compute descriptors
    pcd0_desc = model.compute(pts=pts0, pcd=_pcd0)
    pcd1_desc = model.compute(pts=pts1, pcd=_pcd1)

    # Prepare format for Open3D RANSAC
    pcd0_dsdv = o3d.pipelines.registration.Feature()
    pcd1_dsdv = o3d.pipelines.registration.Feature()

    pcd0_dsdv.data = pcd0_desc.T
    pcd1_dsdv.data = pcd1_desc.T

    _pcd0 = o3d.geometry.PointCloud()
    _pcd0.points = o3d.utility.Vector3dVector(pts0.numpy())
    _pcd1 = o3d.geometry.PointCloud()
    _pcd1.points = o3d.utility.Vector3dVector(pts1.numpy())

    # Apply RANSAC
    est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        _pcd0,
        _pcd1,
        pcd0_dsdv,
        pcd1_dsdv,
        mutual_filter=True,
        max_correspondence_distance=0.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.02)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
    )
    return est_result01

def load_config():
     # Configuration for GeDi
    config = {
        'dim': 32,  # Descriptor output dimension
        'samples_per_batch': 500,  # Batches to process the data on GPU
        'samples_per_patch_lrf': 4000,  # Num. of points to process with LRF
        'samples_per_patch_out': 512,  # Num. of points to sample for PointNet++
        'r_lrf': 0.5,  # LRF radius
        'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'  # Path to checkpoint
    }

    voxel_size = 0.01
    patches_per_pair = 5000

    # Initialize GeDi
    gedi = GeDi(config=config)
    
    return gedi

def data_augmentation(pcd_path, rotation_range, translation_range, noise_std, num_outliers):
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
    pcd.points = o3d.utility.Vector3dVector(np.vstack((np.asarray(pcd.points), outliers)))
    
    return pcd
    

def normalize_point_cloud(pcd):
    # Convert to numpy array
    points = np.asarray(pcd.points)

    # Compute centroid
    centroid = np.mean(points, axis=0)

    # Center the point cloud
    points -= centroid

    # Compute scale
    scale = np.max(np.linalg.norm(points, axis=1))

    # Normalize to fit within unit sphere
    if scale != 0:
        points /= scale

    # Set the points back to the point cloud
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def benchmark(input_dir, target_path, results_path):
    # Load the GeDi model
    model = load_config()

    results = []
    target_pcd = o3d.io.read_point_cloud(target_path)
    
    source_files = os.listdir(input_dir)
    for source in tqdm(source_files, desc="Processing point clouds", unit="file"):
        source_path = os.path.join(input_dir, source)
        source_pcd = data_augmentation(
            source_path, 
            rotation_range=(-np.pi, np.pi), 
            translation_range=(-0.1, 0.1), 
            noise_std=0.01, 
            num_outliers=100
        )
        
        # Compute registration result
        result = compute_registration(source_pcd, target_pcd, model, voxel_size=0.01, patches_per_pair=5000)

        # Create a dictionary for the result
        result_dict = {
            "source_path": source_path,
            "target_path": target_path,
            "fitness": result.fitness,
            "rmse": result.inlier_rmse,
            "transformation": result.transformation.tolist()  # Convert the transformation matrix to a list for JSON compatibility
        }

        # Add the result to the results list
        results.append(result_dict)

    # Save the results to a JSON file
    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved in {results_path}")

################### data analysis ####################
import os
import pandas as pd
import numpy as np
import json
import open3d as o3d
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_results(json_path):
    with open(json_path, 'r') as file:
        results = json.load(file)
    return results

def prepare_data(results):
    # Extract piece number from the source_path
    df = pd.DataFrame(results)
    df['piece_number'] = df['source_path'].apply(lambda x: os.path.basename(x).split('_')[0])
    df['piece_number'] = df['piece_number'].str.replace('.ply', '')
    return df

def plot_comparison(df, target_name):
    # Set the style for seaborn
    sns.set_theme(style="whitegrid")
    
    # Determine the highest fitness and the lowest RMSE
    max_fitness = df['fitness'].max()
    min_rmse = df['rmse'].min()

    # Define color mapping for fitness (highest fitness in green, others in red)
    df['fitness_color'] = np.where(df['fitness'] == max_fitness, 'green', 'red')

    # Define color mapping for RMSE (lowest RMSE in green, others in red)
    df['rmse_color'] = np.where(df['rmse'] == min_rmse, 'green', 'red')

    # Create a bar plot for fitness
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(x='piece_number', y='fitness', data=df, palette=df['fitness_color'].tolist())
    plt.xticks(rotation=60)
    plt.title(f'Fitness Comparison Across Pieces for Target: {target_name}')
    plt.xlabel('Piece Number')
    plt.ylabel('Fitness')
    
    # Highlight the bar for the target piece with underline
    target_index = df[df['piece_number'] == target_name].index
    for i in target_index:
        barplot.patches[i].set_edgecolor('black')
        barplot.patches[i].set_linewidth(2.5)
        barplot.patches[i].set_alpha(0.7)
        # Add underline effect
        plt.gca().add_line(plt.Line2D([barplot.patches[i].get_x() - 0.5, barplot.patches[i].get_x() + barplot.patches[i].get_width() + 0.5],
                                      [-1] * 2,  # Position the underline slightly below the x-axis
                                      color='blue',
                                      linewidth=2))

    # Manually add legend
    handles = [plt.Line2D([0], [0], color='green', lw=4), plt.Line2D([0], [0], color='red', lw=4)]
    plt.legend(handles=handles, title='Fitness Color', loc='upper right', labels=['Highest Fitness', 'Other Fitness'], frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(f'fitness_comparison_{target_name}.png')
    
    # Create a bar plot for RMSE
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(x='piece_number', y='rmse', data=df, palette=df['rmse_color'].tolist())
    plt.xticks(rotation=60)
    plt.title(f'RMSE Comparison Across Pieces for Target: {target_name}')
    plt.xlabel('Piece Number')
    plt.ylabel('RMSE')
    
    # Highlight the bar for the target piece with underline
    target_index = df[df['piece_number'] == target_name].index
    for i in target_index:
        barplot.patches[i].set_edgecolor('black')
        barplot.patches[i].set_linewidth(2.5)
        barplot.patches[i].set_alpha(0.7)
        # Add underline effect
        plt.gca().add_line(plt.Line2D([barplot.patches[i].get_x() - 0.5, barplot.patches[i].get_x() + barplot.patches[i].get_width() + 0.5],
                                      [-1] * 2,  # Position the underline slightly below the x-axis
                                      color='blue',
                                      linewidth=2))

    # Manually add legend
    handles = [plt.Line2D([0], [0], color='green', lw=4), plt.Line2D([0], [0], color='red', lw=4)]
    plt.legend(handles=handles, title='RMSE Color', loc='upper right', labels=['Lowest RMSE', 'Other RMSE'], frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(f'rmse_comparison_{target_name}.png')

def benchmark(input_dir, target_path, results_path):
    # Load the GeDi model
    model = load_config()

    results = []
    target_pcd = o3d.io.read_point_cloud(target_path)
    
    source_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
    for source in tqdm(source_files, desc="Processing point clouds", unit="file"):
        source_path = os.path.join(input_dir, source)
        source_pcd = data_augmentation(
            source_path, 
            rotation_range=(-np.pi, np.pi), 
            translation_range=(-1, 1), 
            noise_std=0.05, 
            num_outliers=200
        )
        
        # Compute registration result
        result = compute_registration(source_pcd, target_pcd, model, voxel_size=0.01, patches_per_pair=5000)

        # Create a dictionary for the result
        result_dict = {
            "source_path": source_path,
            "target_path": target_path,
            "fitness": result.fitness,
            "rmse": result.inlier_rmse,
            "transformation": result.transformation.tolist()  # Convert the transformation matrix to a list for JSON compatibility
        }

        # Add the result to the results list
        results.append(result_dict)

    # Ensure the results directory exists
    results_dir = os.path.dirname(results_path)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}")

    # Save the results to a JSON file
    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved in {results_path}")

if __name__ == "__main__":
    input_dir = "data/10000"
    
    # List all PLY files in the input directory to use as targets
    target_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
    
    for target_file in target_files:
        target_path = os.path.join(input_dir, target_file)
        target_name = target_file.replace('.ply', '')
        results_path = f"data/result/{input_dir.split('/')[-1]}/{target_name}/registration_results.json"
        
        # Run benchmark for each target file
        benchmark(input_dir=input_dir, target_path=target_path, results_path=results_path)
        
        # Load and prepare data
        results = load_results(results_path)
        df = prepare_data(results)
        
        # Plot the comparisons
        plot_comparison(df, target_name)
```