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


def compute_eig_similarity1(eigenvalues1, eigenvalues2):
    scan_eig = np.asarray(eigenvalues1)
    cad_eig = np.asarray(eigenvalues2)
    diff = np.abs(scan_eig - cad_eig) / scan_eig
    score = np.linalg.norm(diff, ord=2)
    d = len(scan_eig)
    return score / np.sqrt(d)


def compute_eig_similarity2(eigenvalues1, eigenvalues2, alpha=1.0):
    scan_eig = np.asarray(eigenvalues1)
    cad_eig = np.asarray(eigenvalues2)
    diff = np.abs(scan_eig - cad_eig) / scan_eig
    base_score = np.sum(diff)
    median_diff = np.median(diff)
    max_diff = np.max(diff)
    penalty = alpha * max(0, max_diff - median_diff)
    score = base_score + penalty
    d = len(scan_eig)
    return score / (d + alpha)


def compute_eig_similarity3(eigenvalues1, eigenvalues2, weights=[1.5, 1.25, 1], alpha=1.0):
    scan_eig = np.asarray(eigenvalues1)
    cad_eig = np.asarray(eigenvalues2)
    diff = np.abs(scan_eig - cad_eig) / scan_eig
    weights = np.array(weights)
    weighted_diff = diff * weights
    base_score = np.sum(weighted_diff)
    median_diff = np.median(weighted_diff)
    max_diff = np.max(weighted_diff)
    penalty = alpha * max(0, max_diff - median_diff)
    score = base_score + penalty
    return score / (np.sum(weights) + alpha)


def apply_filtering(single_scan_df, threshold=1, method="sim_score1"):
    """
    Apply a filtering method to a single scan experiment (1 scan cross with all cads)
    return the list of cads that passed the filtering
    
    return: list of str : ['763620', '763621', '763638', '763640', '763660']
    """
    filtered_index = single_scan_df[single_scan_df[method] > threshold].index
    list_of_str = filtered_index.tolist()
    return list_of_str

def get_rank_coarse(scan, coarse_result_df, metric="score_sim1"):
    """
    Get the rank of a scan in the full dataframe for each metric

    Args:
    full_dataframe: pd.DataFrame (the dataframe containing all the scans coupled with
    the cad and the metrics)

    scan: str (the scan to get the rank for)

    metrics: list of str
    """
    scan_result = coarse_result_df[scan]
    scan_result = scan_result.to_dict()
    scan_result = pd.DataFrame.from_dict(scan_result).T

    summary = {}    
    sorted_df = scan_result.sort_values(by=metric, ascending=True)
    scan_rank = sorted_df.index.get_loc(scan) + 1  # find scan index (1 + idx)

    return scan_rank



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
            points = np.asarray(point_cloud.points)

            if points.shape[0] < 2:
                print(f"Not enough points to compute eigenvalues for {pcd_file}")
                continue

            try:
                # Compute the covariance matrix
                covariance_matrix = np.cov(points.T)
                eigenvalues = np.linalg.eigvals(covariance_matrix)

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

        # Process each scan
        for scan in tqdm(
            [f for f in os.listdir(scan_folder) if f.endswith(".ply")],
            desc="Coarse filtering",
        ):
            scan_result = {}
            scan_number = scan.split(".")[0]
            scan_eig_path = os.path.join(scan_eigenvalues_folder, scan_number + ".npy")
            scan_eigenvalues = np.load(scan_eig_path)

            # Process each CAD file
            for cad in [f for f in os.listdir(cad_folder) if f.endswith(".ply")]:
                cad_number = cad.split(".")[0]
                cad_eig_path = os.path.join(cad_eigenvalues_folder, cad_number + ".npy")
                cad_eigenvalues = np.load(cad_eig_path)
                score_sim1 = compute_eig_similarity1(scan_eigenvalues, cad_eigenvalues)
                score_sim2 = compute_eig_similarity2(scan_eigenvalues, cad_eigenvalues)
                score_sim3 = compute_eig_similarity3(scan_eigenvalues, cad_eigenvalues)

                # Save the similarity scores
                scan_result[cad_number] = {
                    "score_sim1": score_sim1,
                    "score_sim2": score_sim2,
                    "score_sim3": score_sim3,
                }

            exp_results[scan_number] = scan_result

        # Save the results
        df_results = pd.DataFrame.from_dict(exp_results, orient="index")
        saving_path = os.path.join(filtering_coarse_folder)
        df_results.to_json(saving_path)
        print("Coarse filtering results saved here 💾 ===> ", saving_path)

        return df_results


def fine_filtering(scan, candidates, config):
    # Load the config
    scan_preprocess_folder = config["paths"]["scan"]["preprocessed_pcd"] 
    scan_inds_folder = config["paths"]["scan"]["indices"]
    scan_desc_folder = config["paths"]["scan"]["descriptors"]
    cad_preprocess_folder = config["paths"]["cad"]["preprocessed_pcd"]
    cad_inds_folder = config["paths"]["cad"]["indices"]
    cad_desc_folder = config["paths"]["cad"]["descriptors"]
    
    voxel_size = config["model"]["gedi_voxel_size"]
    viz_folder = config["paths"]["results"]["registrations_viz"]
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


def precompute_pipeline(config_path):
    """
    This pipeline preprocesses the scans and CADs, computes the eigenvalues and descriptors for each scan and CAD.
    The pipeline is defined in the config file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

        # Load the GEDI model
        gedi = load_model(config["model"]["gedi_config"])
        patches_per_pair = config["model"]["gedi_patches_per_pair"]
        voxel_size = config["model"]["gedi_voxel_size"]
        scale_factor = config["preprocess_params"]["cad"]["scale_factor"]

        # Preprocessing Scans and CADs
        print("\n\nPreprocessing Scans 🛠️ ... ")
        preprocess_scan(
            source_folder=config["paths"]["scan"]["raw_pcd"],
            target_folder=config["paths"]["scan"]["preprocessed_pcd"],
            patches_per_pair=patches_per_pair,
        )
        print("\n\nPreprocessing CADs  🛠️ ...")
        preprocess_CAD(
            source_folder=config["paths"]["cad"]["raw_pcd"],
            target_folder=config["paths"]["cad"]["preprocessed_pcd"],
            patches_per_pair=patches_per_pair,
            scale_factor=scale_factor,
        )

        # Precomputing scans eigenvalues
        print("\n\nComputing eigenvalues for scans 🛠️ ... ")
        compute_eigenvalues(
            source_folder=config["paths"]["scan"]["preprocessed_pcd"],
            target_folder=config["paths"]["scan"]["eigenvalues"],
        )
        print("\n\nComputing descriptors for cads 🛠️ ...")
        compute_eigenvalues(
            source_folder=config["paths"]["cad"]["preprocessed_pcd"],
            target_folder=config["paths"]["cad"]["eigenvalues"],
        )

        # # Precomputing descriptors
        print("\n\nComputing descriptors for scans  🛠️ ...")
        compute_descriptors(
            source_folder=config["paths"]["scan"]["preprocessed_pcd"],
            desc_target_folder=config["paths"]["scan"]["descriptors"],
            inds_target_folder=config["paths"]["scan"]["indices"],
            patches_per_pair=patches_per_pair,
            voxel_size=voxel_size,
            model=gedi,
        )
        print("\n\nComputing descriptors for cads  🛠️ ...")
        compute_descriptors(
            source_folder=config["paths"]["cad"]["preprocessed_pcd"],
            desc_target_folder=config["paths"]["cad"]["descriptors"],
            inds_target_folder=config["paths"]["cad"]["indices"],
            patches_per_pair=patches_per_pair,
            voxel_size=voxel_size,
            model=gedi,
        )
        
        


def registration_based_pipeline(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        # Compute coarse score of every scan with every cad
        coarse_df = compute_coarse_score(
            scan_folder=config["paths"]["scan"]["preprocessed_pcd"],
            cad_folder=config["paths"]["cad"]["preprocessed_pcd"],
            scan_eigenvalues_folder=config["paths"]["scan"]["eigenvalues"],
            cad_eigenvalues_folder=config["paths"]["cad"]["eigenvalues"],
            filtering_coarse_folder=config["paths"]["results"]["filtering_coarse_overall"],
        )

        # Coarse to fine filtering
        # Can't precompute the fine filtering because it depends on the coarse filtering
        rank_coarse_results = {}
        fine_results = {}
        
        # Use the coarse score to filter the scans
        for scan in coarse_df.index:
            print(f"Filtering scan experiment {scan}")
            rank_coarse_results[scan] = {}
            coarse_scan_results = coarse_df[scan]
            coarse_scan_results = coarse_scan_results.to_dict()
            coarse_scan_results = pd.DataFrame.from_dict(coarse_scan_results).T

            # to make test, comment the fine filtering, and add as many metrics as you need
            for method in ["score_sim1", "score_sim2", "score_sim3"]:
                # Sort and apply the filtering
                filtered_candidates = apply_filtering(coarse_scan_results, method=method, threshold=1)
                rank = get_rank_coarse(scan, coarse_df, metric=method)
                # print(rank)
                rank_coarse_results[scan][method] = rank
                
                # Fine filtering...
                # fine_result = fine_filtering(scan, filtered_candidates, config)
                # fine_results[scan] = fine_result
                
                
            
            
        # log scan experiment ranks
        coarse_rank_df = pd.DataFrame.from_dict(rank_coarse_results, orient="index")
        coarse_rank_df.to_json(config["paths"]["results"]["filtering_coarse_rank"])
        print("Ranking of scan experiments saved here 💾 ===> ", config["paths"]["results"]["filtering_coarse_rank"])
        # TODO : compute mean and std of the ranks of each method
        
        fine_results_path = config["paths"]["results"]["filtering_fine"]
        path_checker1(fine_results_path)
        file_path = os.path.join(fine_results_path, "fine_results.json")
        pd.DataFrame.from_dict(fine_results).to_json(config["paths"]["results"]["filtering_fine"])
        print("Fine filtering results saved here 💾 ===> ", config["paths"]["results"]["filtering_fine"])

        
        
        
        
if __name__ == "__main__":
    config = "/app/bindmount/gedi_data_2/config.yaml"
    # precompute_pipeline(config)
    registration_based_pipeline(config)
