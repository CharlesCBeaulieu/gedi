import os
import open3d as o3d
import numpy as np
import yaml
from tqdm import tqdm
from gedi import GeDi
import torch
import pandas as pd
import copy


def path_checker(source_folder, *target_folders):
    if not os.path.exists(source_folder):
        print(f"Source folder {source_folder} does not exist")
        return False
    for folder in target_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder {folder}")
    return True

def path_checker1(*target_folders):
    for folder in target_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder {folder}")
    return True


def load_model(config):
    return GeDi(config)


def get_max_eigen_from_folder(folder_path):
    max_val = None
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            arr = np.load(file_path)
            current_max = np.max(arr)
            if max_val is None or current_max > max_val:
                max_val = current_max
    return max_val


# mean : 6.55 ; std : 8.78
# norme L1, (faudrait tester la norme L2)
def compute_eig_similarity1(scan_eig, cad_eig):
    diff = np.abs(scan_eig - cad_eig)
    score = np.sum(diff)
    return score

def compute_eig_similarity2(scan_eig, cad_eig):
    diff = scan_eig - cad_eig
    score = np.linalg.norm(diff, ord=2)
    return score

# mean : 7.42 ; std : 9.33
# This one game me an averege rank of 7.42 and standard deviation of 9.33
# def compute_eig_similarity1(scan_eig, cad_eig, penalty=True, dim_weight=[1, 1, 1], max_eigenvalue=1, shift_penalty_factor=2, shift_threshold=0.3):
#     diff = np.abs(scan_eig - cad_eig)
#     rmse = np.sqrt(np.mean(diff ** 2))
#     return rmse


# c'est la metric utilisez dans la presentation du 10 avril
# def compute_eig_similarity2(scan_eig, cad_eig):
#     diff = np.abs(scan_eig - cad_eig)
#     penalty = np.std(diff)
#     return np.sum(diff) + penalty


def compute_eig_similarity3(scan_eig, cad_eig, scan_points=None, cad_points=None, scan_number=None, cad_number=None):
    # Normalized score (does not take the size of the object into account) + we add the difference of size
    norm_scan = scan_eig / scan_eig.sum()
    norm_cad  = cad_eig / cad_eig.sum()
    score_norm = np.sum(np.abs(norm_scan - norm_cad))
    
    scan_diag = np.linalg.norm(scan_points.max(axis=0) - scan_points.min(axis=0))
    cad_diag  = np.linalg.norm(cad_points.max(axis=0) - cad_points.min(axis=0))
    scale_diff = np.abs(scan_diag - cad_diag)
    
    w = 1 
    
    if True:
        print(f"score_norm : {score_norm}, scale_diff : {scale_diff}")

    return score_norm + w * scale_diff


# def compute_eig_similarity2(scan_eig, cad_eig):
#     # Normalized score (does not take the size of the object into account)
#     norm_scan = scan_eig / scan_eig.sum()
#     norm_cad  = cad_eig / cad_eig.sum()
#     score_norm = np.sum(np.abs(norm_scan - norm_cad))
    
#     # this take the size
#     size_scan = scan_eig.sum()
#     size_cad  = cad_eig.sum()
#     scale_diff = np.abs(size_scan - size_cad) / size_scan
    
#     w = 1  # weight factor for the scale difference
    
#     print(score_norm + w * scale_diff)

#     return score_norm + w * scale_diff

# def compute_eig_similarity2(scan_eig, cad_eig):
#     """Anisotropy ratio"""
#     ratios_scan = np.array([scan_eig[0] / scan_eig[1], scan_eig[1] / scan_eig[2]])
#     ratios_cad  = np.array([cad_eig[0] / cad_eig[1], cad_eig[1] / cad_eig[2]])
#     score_ratios = np.sum(np.abs(ratios_scan - ratios_cad))
#     return score_ratios



    

def apply_filtering(single_scan_df, threshold=False, method="sim_score1"):
    """
    Apply a filtering method to a single scan experiment (1 scan cross with all cads)
    return the list of cads that passed the filtering
    
    Threshold is better, but less convenient to tune. But in fact the number of similar cads 
    in the dataset can highly affect the Ktop but not the threshold.
    
    return: list of str : ['763620', '763621', '763638', '763640', '763660']
    """
    filtered_index = single_scan_df[single_scan_df[method] < threshold].index
        
    return filtered_index.tolist()



def get_rank_coarse(scan, coarse_result_df, metric="score_sim1"):
    """
    Get the rank of a scan in the full dataframe for each metric

    Args:
    full_dataframe: pd.DataFrame (the dataframe containing all the scans coupled with
    the cad and the metrics)

    scan: str (the scan to get the rank for)

    metrics: list of str
    """
    scan_result = coarse_result_df.loc[scan]
    scan_result = scan_result.to_dict()
    scan_result = pd.DataFrame.from_dict(scan_result).T

    sorted_df = scan_result.sort_values(by=metric, ascending=True)
    scan_rank = sorted_df.index.get_loc(scan) + 1  # find scan index (1 + idx)
    
    print(f"Rank : {scan_rank}")
    print(f"Score : {scan_result.loc[scan][metric]}")
    
    return scan_rank



def get_rank_fine(scan_key, registration_result):
    df = pd.DataFrame.from_dict(registration_result, orient="index")
    df_sorted = df.sort_values(by="ransac_fitness", ascending=False)
    
    try:
        rank = df_sorted.index.get_loc(scan_key) + 1 
    except KeyError:
        rank = None  # Si le scan n'est pas trouvé, on retourne None
        
    return rank



def sort_by_metric_coarse(coarse_result_df, metrics=["score_sim1", "score_sim2"]):
    """
    Get the rank of each scan in the full dataframe for each metric

    Args:
    full_dataframe: pd.DataFrame (the dataframe containing all the scans coupled with
    the cad and the metrics)

    metrics: list of str
    """
    summary = {}

    for scan in coarse_result_df:
        scan_result = coarse_result_df[scan]
        scan_result = scan_result.to_dict()
        scan_result = pd.DataFrame.from_dict(scan_result).T
        summary[scan] = {}

        # Get the rank of the scan for each metric
        for metric in metrics:
            sorted_df = scan_result.sort_values(by=metric, ascending=True)
            scan_index = sorted_df.index.get_loc(scan) + 1  # find scan index (1 + idx)
            print(f"Scan {scan} is ranked {scan_index} in {metric}")
            summary[scan][metric] = scan_index

    return summary


def preprocess_scan(source_folder, target_folder, patches_per_pair):
    # check if the paths are valid
    if path_checker(source_folder, target_folder):
        for scan in tqdm(
            [f for f in os.listdir(source_folder) if f.endswith(".ply")],
            desc="Preprocessing scans",
        ):
            # Load the scan
            scan_path = os.path.join(source_folder, scan)
            pcd = o3d.io.read_point_cloud(scan_path)
            points = np.asarray(pcd.points)

            # Check if the point cloud is big enough for the model
            if points.shape[0] < patches_per_pair:
                print(f"Not enough points : {scan}")
                print(f"Skipping {scan}")
                continue

            # Center the scan
            original_centroid = np.mean(points, axis=0)
            centered_points = points - original_centroid
            pcd.points = o3d.utility.Vector3dVector(centered_points)

            # Save the centered scan
            centered_scan_path = os.path.join(target_folder, scan)
            o3d.io.write_point_cloud(centered_scan_path, pcd)

        print("Preprocessed point cloud saved here 💾 ===> ", target_folder)


def preprocess_CAD(source_folder, target_folder, patches_per_pair, scale_factor=0.001):
    # check if the paths are valid
    if path_checker(source_folder, target_folder):
        for cad in tqdm(
            [f for f in os.listdir(source_folder) if f.endswith(".ply")],
            desc="Preprocessing CADs",
        ):
            # Load the CAD
            cad_path = os.path.join(source_folder, cad)
            cad_pcd = o3d.io.read_point_cloud(cad_path)
            points = np.asarray(cad_pcd.points)

            # Check if the point cloud is big enough for the model
            if points.shape[0] < patches_per_pair:
                print(f"Not enough points : {cad}")
                print(f"Skipping {cad}")
                continue

            # Scale the CAD
            scaled_points = points * scale_factor

            # center the CAD
            original_centroid = np.mean(scaled_points, axis=0)
            centered_points = scaled_points - original_centroid
            cad_pcd.points = o3d.utility.Vector3dVector(centered_points)

            # Save the centered CAD
            centered_cad_path = os.path.join(target_folder, cad)
            o3d.io.write_point_cloud(centered_cad_path, cad_pcd)

        print("Preprocessed CADs saved here 💾 ===> ", target_folder)


def compute_eigenvalues(source_folder, target_folder):
    # check if the paths are valid
    if path_checker(source_folder, target_folder):
        for pcd_file in tqdm(os.listdir(source_folder), desc="Computing eigenvalues"):
            # Load the scan
            pcd_file_path = os.path.join(source_folder, pcd_file)
            point_cloud = o3d.io.read_point_cloud(pcd_file_path)
            
            # voxel downsample the point cloud to avoid density issues
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.001) # 0.1cm
            points = np.asarray(point_cloud.points)

            if points.shape[0] < 2:
                print(f"Not enough points to compute eigenvalues for {pcd_file}")
                continue

            try:
                # Compute the covariance matrix
                covariance_matrix = np.cov(points.T)
                eigenvalues = np.linalg.eigvals(covariance_matrix)
                
                # Sort the eigenvalues biggest to smallest
                eigenvalues = np.sort(eigenvalues)[::-1]

                # Save the eigenvalues
                pcd_file = pcd_file.split(".")[0]
                pcd_file += ".npy"
                eigenvalues_path = os.path.join(target_folder, pcd_file)
                np.save(eigenvalues_path, eigenvalues)
            except:
                print(f"Error computing eigenvalues for {pcd_file}")
                continue

        print("Eigenvalues saved here 💾 ===>", target_folder)


def compute_descriptors(
    source_folder,
    desc_target_folder,
    inds_target_folder,
    patches_per_pair,
    voxel_size,
    model="gedi",
):
    if path_checker(source_folder, desc_target_folder, inds_target_folder):
        for pcd_file in tqdm(
            [f for f in os.listdir(source_folder) if f.endswith(".ply")],
            desc="Computing descriptors",
        ):
            # Load the scan
            pcd_file_path = os.path.join(source_folder, pcd_file)
            point_cloud = o3d.io.read_point_cloud(pcd_file_path)
            points = np.asarray(point_cloud.points)

            # Check if the point cloud is empty
            if points.shape[0] == 0:
                raise ValueError("The point cloud contains no points.")
            if points.shape[0] < patches_per_pair:
                # skip the point cloud
                print(f"Not enough points to compute descriptors for {pcd_file}")
                # it will be needed to remove those scan or cads from the dataset
                continue

            # Sample points
            inds = np.random.choice(points.shape[0], patches_per_pair, replace=False)
            pts_sample = points[inds]
            pts_tensor = torch.tensor(pts_sample).float()

            # Compute the descriptor
            pcd_down = point_cloud.voxel_down_sample(voxel_size=voxel_size)
            pcd_down.estimate_normals()
            down_points = np.asarray(pcd_down.points)
            pcd_down_tensor = torch.tensor(down_points).float()
            descriptor = model.compute(pts=pts_tensor, pcd=pcd_down_tensor)

            # Save the descriptor
            pcd_file = pcd_file.split(".")[0]
            pcd_file += ".npy"
            descriptor_path = os.path.join(desc_target_folder, pcd_file)
            np.save(descriptor_path, descriptor)

            # Save the indices
            inds_path = os.path.join(inds_target_folder, pcd_file)
            np.save(inds_path, inds)

        print("Descriptors saved here 💾 ===> ", desc_target_folder)
        print("Indices saved here 💾 ===> ", inds_target_folder)


def compute_coarse_score(
    scan_folder,
    cad_folder,
    scan_eigenvalues_folder,
    cad_eigenvalues_folder,
    filtering_coarse_folder,
    threshold=1,
):
    # Ensure paths exist
    # Check that the required folders exist
    if path_checker(
        scan_folder, cad_folder, scan_eigenvalues_folder, cad_eigenvalues_folder
    ):
        exp_results = {}
        max_eigenvalue = get_max_eigen_from_folder(cad_eigenvalues_folder)
        # Process each scan
        for scan in tqdm(
            [f for f in os.listdir(scan_folder) if f.endswith(".ply")],
            desc="Computing coarse scores",
        ):
            scan_result = {}
            scan_number = scan.split(".")[0]
            scan_eig_path = os.path.join(scan_eigenvalues_folder, scan_number + ".npy")
            scan_eigenvalues = np.load(scan_eig_path)
            
            # get the scan points
            scan_points_path = os.path.join(scan_folder, scan_number + ".ply")
            pcd_scan = o3d.io.read_point_cloud(scan_points_path)
            points_scan = np.asarray(pcd_scan.points)

            # Process each CAD file
            for cad in [f for f in os.listdir(cad_folder) if f.endswith(".ply")]:
                cad_number = cad.split(".")[0]
                cad_eig_path = os.path.join(cad_eigenvalues_folder, cad_number + ".npy")
                cad_eigenvalues = np.load(cad_eig_path)
                
                # load the cad points
                cad_points_path = os.path.join(cad_folder, cad_number + ".ply")
                pcd_cad = o3d.io.read_point_cloud(cad_points_path)
                points_cad = np.asarray(pcd_cad.points)
                
                # plot_scan_cad(points_scan, points_cad)
                
                score_sim1 = compute_eig_similarity1(scan_eigenvalues, cad_eigenvalues)
                score_sim2 = compute_eig_similarity2(scan_eigenvalues, cad_eigenvalues)
                score_sim3 = compute_eig_similarity3(scan_eigenvalues, cad_eigenvalues, scan_points=points_scan, cad_points=points_cad, scan_number=scan_number, cad_number=cad_number)
                
                # Save the similarity scores
                scan_result[cad_number] = {
                    "score_sim1": score_sim1,
                    "score_sim2": score_sim2,
                    "score_sim3": score_sim3,
                }

            exp_results[scan_number] = scan_result

        # Save the results
        df_results = pd.DataFrame.from_dict(exp_results, orient="index")
        print("Coarse filtering results saved here 💾 ===> ", filtering_coarse_folder)

        return df_results


def plot_scan_cad(scan_points, cad_points):
    """
    Plot the scan and cad points in 3D
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scan_points[:, 0], scan_points[:, 1], scan_points[:, 2], c='r', marker='o')
    ax.scatter(cad_points[:, 0], cad_points[:, 1], cad_points[:, 2], c='b', marker='^')
    plt.savefig("scan_cad.png")



def fine_filtering(scan, candidates, config):
    # Load the config
    base = config["base"]
    scan_preprocess_folder = os.path.join(base, config["paths"]["scan"]["preprocessed_pcd"])
    scan_inds_folder = os.path.join(base, config["paths"]["scan"]["indices"])
    scan_desc_folder = os.path.join(base, config["paths"]["scan"]["descriptors"])
    cad_preprocess_folder = os.path.join(base, config["paths"]["cad"]["preprocessed_pcd"])
    cad_inds_folder = os.path.join(base, config["paths"]["cad"]["indices"])
    cad_desc_folder = os.path.join(base, config["paths"]["cad"]["descriptors"])
    
    voxel_size = config["model"]["gedi_voxel_size"]
    viz_folder = os.path.join(base, config["paths"]["results"]["registrations_viz"])
    path_checker1(viz_folder)
    
    # Load the scan
    scan_path = os.path.join(scan_preprocess_folder, scan + ".ply")
    pcd0 = o3d.io.read_point_cloud(scan_path)
    
    # load the scan indices
    inds0 = np.load(os.path.join(scan_inds_folder, scan + ".npy"))
    pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()

    # downsample the scan
    pcd0 = pcd0.voxel_down_sample(voxel_size)
    _pcd0 = o3d.geometry.PointCloud()
    _pcd0.points = o3d.utility.Vector3dVector(pts0)

    # load the scan descriptor
    pcd0_desc = np.load(os.path.join(scan_desc_folder, scan + ".npy"))
    pcd0_dsdv = o3d.pipelines.registration.Feature()
    pcd0_dsdv.data = pcd0_desc.T

    registration_result = {}

    # for each CAD
    for idx, cad in enumerate(tqdm(candidates, desc="Processing registrations for fine filtering")):
        # Load the CAD
        cad_path = os.path.join(cad_preprocess_folder, cad + ".ply")
        pcd1 = o3d.io.read_point_cloud(cad_path)

        # load the CAD indices
        inds1 = np.load(os.path.join(cad_inds_folder, cad + ".npy"))
        pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

        # downsample the CAD
        pcd1 = pcd1.voxel_down_sample(voxel_size)
        _pcd1 = o3d.geometry.PointCloud()
        _pcd1.points = o3d.utility.Vector3dVector(pts1)

        # load the CAD descriptor
        pcd1_desc = np.load(os.path.join(cad_desc_folder, cad + ".npy"))
        pcd1_dsdv = o3d.pipelines.registration.Feature()
        pcd1_dsdv.data = pcd1_desc.T

        if _pcd0.is_empty() or _pcd1.is_empty():
            raise ValueError("Empty point cloud for scan or CAD.")

        est_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            _pcd0,
            _pcd1,
            pcd0_dsdv,
            pcd1_dsdv,
            mutual_filter=True,
            max_correspondence_distance=0.02,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.02),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
        )

        registration_result[cad] = {
            "ransac_fitness": est_result.fitness,
            "ransac_inlier_rmse": est_result.inlier_rmse,
            "ransac_transformation": est_result.transformation,
        }

        # Visualize the alignment
        pcd0_copy = copy.deepcopy(_pcd0)
        pcd0_copy.paint_uniform_color([0, 0.651, 0.929])
        pcd0_copy.transform(est_result.transformation)
        pcd1.paint_uniform_color([1, 0.706, 0])
        combined_ransac = pcd0_copy + pcd1
        # create a folder for each scan
        scan_folder = os.path.join(viz_folder, scan)
        path_checker1(scan_folder)
        ransac_filename = os.path.join(scan_folder, f"scan{scan}_cad{cad}_aligned.ply")
        o3d.io.write_point_cloud(ransac_filename, combined_ransac)
        
    return registration_result


        
if __name__ == "__main__":
    pass
