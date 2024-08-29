---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: venv
    language: python
    name: python3
---

```python
import open3d as o3d
import numpy as np

def get_info_pcd(pcd : o3d.geometry.PointCloud):
    points = np.asarray(pcd.points)
    
    # Compute min max
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    print(f"x_min: {x_min}, x_max: {x_max}")
    print(f"y_min: {y_min}, y_max: {y_max}")
    print(f"z_min: {z_min}, z_max: {z_max}")

    number_of_points = len(points)
    print(f"Number of points : {number_of_points}")
    
    return x_min, x_max, y_min, y_max, z_min, z_max, number_of_points
```

# Analyse des différents type de pcd


### Analyse des samples de 3DMatch

```python
open3d_exemple1 = o3d.io.read_point_cloud("../data/assets/threed_match_7-scenes-redkitchen_cloud_bin_0.ply")
open3d_exemple2 = o3d.io.read_point_cloud("../data/assets/threed_match_7-scenes-redkitchen_cloud_bin_5.ply")

_ = get_info_pcd(open3d_exemple1)
print("\n")
_ = get_info_pcd(open3d_exemple2)
```

### Analyse des Scan

```python
scanSample1 = o3d.io.read_point_cloud("../data/real/Scan/segment_1.ply")
scanSample2 = o3d.io.read_point_cloud("../data/real/Scan/segment_2.ply")
scanSample3 = o3d.io.read_point_cloud("../data/real/Scan/segment_3.ply")

_ = get_info_pcd(scanSample1)
print("\n")
_ = get_info_pcd(scanSample2)
print("\n")
_ = get_info_pcd(scanSample3)
```

### Analyse des CAD

```python
cadSample1 = o3d.io.read_point_cloud("../data/real/CAD/779632.ply")
cadSample2 = o3d.io.read_point_cloud("../data/real/CAD/785582.ply")
cadSample3 = o3d.io.read_point_cloud("../data/real/CAD/793998.ply")

_ = get_info_pcd(cadSample1)
print("\n")
_ = get_info_pcd(cadSample2)
print("\n")
_ = get_info_pcd(cadSample3)
```

# Observation

- Le domaine des scans ressemble beaucoup a celui des samples de 3DMatch utilisés pour l'entrainement.  

- Le domaine des CAD est très différent. 


# Normalisation

```python
def center_and_normalize_pcd(pcd: o3d.geometry.PointCloud):
    """ This function centers the point cloud and normalizes it into a unit cube while keeping the aspect ratio """
    points = np.asarray(pcd.points)
    
    # Calculate the centroid of the point cloud
    centroid = np.mean(points, axis=0)
    
    # Center the point cloud
    points -= centroid
    
    # Calculate the min and max values after centering
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    
    # Calculate the scale factor based on the largest dimension
    max_scale = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Normalize the point cloud into a unit cube while keeping the aspect ratio
    points /= max_scale
    
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd
```

```python
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

def display_point_cloud(pcd: o3d.geometry.PointCloud):
    """ This function displays a point cloud using matplotlib with equal axes """
    points = np.asarray(pcd.points)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set the aspect ratio to be equal
    max_range = np.array([points[:, 0].max() - points[:, 0].min(), 
                          points[:, 1].max() - points[:, 1].min(), 
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()
    
    
def display_point_clouds_side_by_side(source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud):
    """ This function displays two point clouds side by side using matplotlib with equal axes """
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    
    fig = plt.figure(figsize=(12, 6))
    
    # Plot source point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], s=1, c=[[0, 0.651, 0.929]])
    ax1.set_title('Source Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Set the aspect ratio to be equal for source point cloud
    max_range = np.array([source_points[:, 0].max() - source_points[:, 0].min(), 
                          source_points[:, 1].max() - source_points[:, 1].min(), 
                          source_points[:, 2].max() - source_points[:, 2].min()]).max() / 2.0

    mid_x = (source_points[:, 0].max() + source_points[:, 0].min()) * 0.5
    mid_y = (source_points[:, 1].max() + source_points[:, 1].min()) * 0.5
    mid_z = (source_points[:, 2].max() + source_points[:, 2].min()) * 0.5

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Plot target point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], s=1, c=[[1, 0.706, 0]])
    ax2.set_title('Target Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Set the aspect ratio to be equal for target point cloud
    max_range = np.array([target_points[:, 0].max() - target_points[:, 0].min(), 
                          target_points[:, 1].max() - target_points[:, 1].min(), 
                          target_points[:, 2].max() - target_points[:, 2].min()]).max() / 2.0

    mid_x = (target_points[:, 0].max() + target_points[:, 0].min()) * 0.5
    mid_y = (target_points[:, 1].max() + target_points[:, 1].min()) * 0.5
    mid_z = (target_points[:, 2].max() + target_points[:, 2].min()) * 0.5

    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()
```

# Vizualisation du domain gap

Affichage des 3 couples scan/CAD

On peut bien voir le domain gab, l'echelle n'est pas respecté

```python
display_point_clouds_side_by_side(scanSample1, cadSample1)
display_point_clouds_side_by_side(scanSample2, cadSample2)
display_point_clouds_side_by_side(scanSample3, cadSample3)
```

```python
sources = [scanSample1, scanSample2, scanSample3]
targets = [cadSample1, cadSample2, cadSample3]

for source, target in zip(sources, targets):
    source = center_and_normalize_pcd(source)
    target = center_and_normalize_pcd(target)
    
    display_point_clouds_side_by_side(source, target)
```

# Matching

```python
def concatenate_point_clouds(pcd0, pcd1):
    """ This function concatenates two point clouds and their color information """
    points0 = np.asarray(pcd0.points)
    colors0 = np.asarray(pcd0.colors)
    
    points1 = np.asarray(pcd1.points)
    colors1 = np.asarray(pcd1.colors)
    
    points = np.concatenate((points0, points1), axis=0)
    colors = np.concatenate((colors0, colors1), axis=0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd
```

```python
# Add the directory containing the notebook to the Python path
import sys
import os 

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname("evaluate_data.ipynb"), 'app')))
import numpy as np
import open3d as o3d
import sys
import os
import torch
from gedi import GeDi
import posixpath

def demo_code(source, target): 
    '''
    demo to show the registration between two point clouds using GeDi descriptors
    - the first visualisation shows the two point clouds in their original reference frame
    - the second visualisation show point cloud 0 transformed in the reference frame of point cloud 1
    '''

    config = {'dim': 32,                                            # descriptor output dimension
            'samples_per_batch': 500,                             # batches to process the data on GPU
            'samples_per_patch_lrf': 4000,                        # num. of point to process with LRF
            'samples_per_patch_out': 512,                         # num. of points to sample for pointnet++
            'r_lrf': .5,                                          # LRF radius
            'fchkpt_gedi_net': '../data/chkpts/3dmatch/chkpt.tar'}   # path to checkpoint

    voxel_size = .01
    patches_per_pair = 5000

    # initialising class
    gedi = GeDi(config=config)

    # getting a pair of point clouds
    pcd0 = source
    pcd1 = target

    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])

    # estimating normals (only for visualisation)
    pcd0.estimate_normals()
    pcd1.estimate_normals()

    display_point_clouds_side_by_side(pcd0, pcd1)

    # randomly sampling some points from the point cloud
    inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=False)
    inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches_per_pair, replace=False)

    pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
    pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

    # applying voxelisation to the point cloud
    pcd0 = pcd0.voxel_down_sample(voxel_size)
    pcd1 = pcd1.voxel_down_sample(voxel_size)

    _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
    _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

    # computing descriptors
    pcd0_desc = gedi.compute(pts=pts0, pcd=_pcd0)
    pcd1_desc = gedi.compute(pts=pts1, pcd=_pcd1)
    
    # preparing format for open3d ransac
    pcd0_dsdv = o3d.pipelines.registration.Feature()
    pcd1_dsdv = o3d.pipelines.registration.Feature()

    pcd0_dsdv.data = pcd0_desc.T
    pcd1_dsdv.data = pcd1_desc.T

    _pcd0 = o3d.geometry.PointCloud()
    _pcd0.points = o3d.utility.Vector3dVector(pts0)
    _pcd1 = o3d.geometry.PointCloud()
    _pcd1.points = o3d.utility.Vector3dVector(pts1)

    # applying ransac
    est_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        _pcd0,
        _pcd1,
        pcd0_dsdv,
        pcd1_dsdv,
        mutual_filter=True,
        max_correspondence_distance=.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

    # applying estimated transformation
    pcd0.transform(est_result.transformation)

    # print("Transformation is:")
    # for row in est_result.transformation:
    #     print(row)

    display_point_clouds_side_by_side(pcd0, pcd1)
    combined_pcd = concatenate_point_clouds(pcd0, pcd1)
    
    return combined_pcd, est_result

```

```python
source = scanSample3
target = cadSample3
saving_path = "../data/real_output/gedi_scan2CAD_drawer_combined.ply"

resgister_pcd, est_result = demo_code(source=source, target=target)
o3d.io.write_point_cloud(saving_path, resgister_pcd)
print(f"Combined point cloud saved at {saving_path}")
```

# Align the merge by camera result

```python
source = o3d.io.read_point_cloud("../data/real_rack_with_pieces/merged_by_camera0.ply")
target = o3d.io.read_point_cloud("../data/real_rack_with_pieces/merged_by_camera1.ply")

saving_path = "../data/real_rack_with_pieces/output/gedi_camera0_to_camera1.ply"

resgister_pcd, est_result = demo_code(source=source, target=target)
o3d.io.write_point_cloud(saving_path, resgister_pcd)
print(f"Combined point cloud saved at {saving_path}")
```

result is very very impressive. 


# Try to align the raw scan without any processing

```python
# first step, we need to align 2 of them together, without preprocessing it's a big challenge

source = o3d.io.read_point_cloud("../data/raw_scans/camera1/scan_0.ply")
target = o3d.io.read_point_cloud("../data/raw_scans/camera1/scan_1.ply")

saving_path = "../data/raw_scans/output/scan0_and_scan1.ply"

resgister_pcd, est_result = demo_code(source=source, target=target)
o3d.io.write_point_cloud(saving_path, resgister_pcd)
print(f"Combined point cloud saved at {saving_path}")
```

# Try to align them all together

to align together we need to align in cascade style, so

1. scan0 register to scan1

2. concatenate register scan0 and scan1

3. ther result register to scan 2

4. ... 

```python
# Take all the ply files in the folder
scans = []
for file in os.listdir("../data/raw_scans/camera1/"):
    if file.endswith(".ply"):
        scans.append(o3d.io.read_point_cloud(os.path.join("../data/raw_scans/camera1/", file)))

# for scan in scans:
#     scan = center_and_normalize_pcd(scan)

# Register and concatenate point clouds in a cascade style
for i in range(len(scans) - 1):
    source = scans[i]
    target = scans[i + 1]
    combined_registered_pcd, est_result = demo_code(source=source, target=target)
    scans[i + 1] = combined_registered_pcd
    o3d.io.write_point_cloud(f"../data/raw_scans/output/camera1/exp3/scan{i}_to_scan{i+1}.ply", combined_registered_pcd)

# Save the final concatenated point cloud
save_path = "../data/raw_scans/output/camera1/exp3/scan0_to_scan_last.ply"
o3d.io.write_point_cloud(save_path, scans[-1])
    
    
```

# Next Week

- essayer d'utiliser la pose pour crop le plus de bruit possible, background etc. 
- refaire des test pour voir si cela améliore le tout
- coder une version du pipeline complet et eventuellement le faire un .py 

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

# Code for Benchmark with Real Data

<!-- #region -->
### data structure

```markdown
Projet/
│
├── data/                   
│   ├── real/
│   │   ├── CAD/
│   │   │   ├── #piece.ply
│   │   │   ├── #piece.ply
│   │   │   ├── ...
│   │   ├── Scan/  
│   │   │   ├── #piece.ply
│   │   │   ├── #piece.ply
│   │   │   ├── ...  
│   │      
│   └── synthetic/
│   │   ├── 1000/
│   │   │   ├── #piece.ply
│   │   │   ├── #piece.ply
│   │   │   ├── ...
│   │   │   
│   │   ├── 10000/
│   │   │   ├── #piece.ply
│   │   │   ├── #piece.ply
│   │   │   ├── ...
```

There need to be 2 types of benchmark, one for the real data that match scan with CAD, and the synthetic that match same pieces but with augmentation

TODO : try to match scan with scan but with data augmentation (so we see if the problem is the domain gap between scan and cad, or if it's the scan quality.)
<!-- #endregion -->

```python

```

```python

```
