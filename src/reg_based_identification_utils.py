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

def compute_eig_similarity1(scan_eig, cad_eig, penalty=False, dim_weight=[1, 1, 1], max_eigenvalue=1):
    diff = np.abs(scan_eig - cad_eig) / max(scan_eig)
    diff = diff[:3] * dim_weight
    penalty_val = np.std(scan_eig) if penalty else 0
    return np.sum(diff) + penalty_val


def compute_eig_similarity2(scan_eig, cad_eig, penalty=True, dim_weight=[1, 0.5, 0.25], max_eigenvalue=1):
    diff = np.abs(scan_eig - cad_eig) / max_eigenvalue
    diff = diff[:3] * dim_weight
    penalty_val = np.std(scan_eig) if penalty else 0
    return np.sum(diff) + penalty_val


def compute_eig_similarity3(eigenvalues1, eigenvalues2, weights=[1, 0.5, 0.25], alpha=1.0):
    scan_eig = np.asarray(eigenvalues1)
    cad_eig = np.asarray(eigenvalues2)
    score = np.dot(scan_eig, cad_eig) / max(cad_eig)
    return score


def apply_filtering(single_scan_df, threshold=1, method="sim_score1"):
    """
    Apply a filtering method to a single scan experiment (1 scan cross with all cads)
    return the list of cads that passed the filtering
    
    return: list of str : ['763620', '763621', '763638', '763640', '763660']
    """
    filtered_index = single_scan_df[single_scan_df[method] < threshold].index
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
                score_sim1 = compute_eig_similarity1(scan_eigenvalues, cad_eigenvalues, max_eigenvalue=max_eigenvalue)
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
        print("Coarse filtering results saved here 💾 ===> ", filtering_coarse_folder)

        return df_results


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


def precompute_pipeline(config_path):
    """
    This pipeline preprocesses the scans and CADs, computes the eigenvalues and descriptors for each scan and CAD.
    The pipeline is defined in the config file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
        # Get the base path
        base = config["base"]
        gedi = load_model(config["model"]["gedi_config"])
        
        # For scans
        raw_scan_folder = os.path.join(base, config["paths"]["scan"]["raw_pcd"].lstrip("/"))
        preprocessed_scan_folder = os.path.join(base, config["paths"]["scan"]["preprocessed_pcd"].lstrip("/"))
        scan_eigenvalues_folder = os.path.join(base, config["paths"]["scan"]["eigenvalues"].lstrip("/"))
        scan_descriptors_folder = os.path.join(base, config["paths"]["scan"]["descriptors"].lstrip("/"))
        scan_indices_folder = os.path.join(base, config["paths"]["scan"]["indices"].lstrip("/"))

        # For CADs
        raw_cad_folder = os.path.join(base, config["paths"]["cad"]["raw_pcd"].lstrip("/"))
        preprocessed_cad_folder = os.path.join(base, config["paths"]["cad"]["preprocessed_pcd"].lstrip("/"))
        cad_eigenvalues_folder = os.path.join(base, config["paths"]["cad"]["eigenvalues"].lstrip("/"))
        cad_descriptors_folder = os.path.join(base, config["paths"]["cad"]["descriptors"].lstrip("/"))
        cad_indices_folder = os.path.join(base, config["paths"]["cad"]["indices"].lstrip("/"))
        
        print("\n\nPreprocessing scans 🛠️ ...")
        preprocess_scan(source_folder=raw_scan_folder,
                        target_folder=preprocessed_scan_folder,
                        patches_per_pair=config["model"]["gedi_patches_per_pair"])
        
        print("\n\nPreprocessing cads  🛠️ ...")
        preprocess_CAD(source_folder=raw_cad_folder,
                    target_folder=preprocessed_cad_folder,
                    patches_per_pair=config["model"]["gedi_patches_per_pair"],
                    scale_factor=config["preprocess_params"]["cad"]["scale_factor"])
        
        print("\n\nComputing eigenvalues for scans 🛠️ ... ")
        compute_eigenvalues(source_folder=preprocessed_scan_folder,
                            target_folder=scan_eigenvalues_folder)
        
        print("\n\nComputing eigenvalues for cads 🛠️ ... ")
        compute_eigenvalues(source_folder=preprocessed_cad_folder,
                            target_folder=cad_eigenvalues_folder)
        
        print("\n\nComputing descriptors for scans  🛠️ ...")
        compute_descriptors(source_folder=preprocessed_scan_folder,
                            desc_target_folder=scan_descriptors_folder,
                            inds_target_folder=scan_indices_folder,
                            patches_per_pair=config["model"]["gedi_patches_per_pair"],
                            voxel_size=config["model"]["gedi_voxel_size"],
                            model=load_model(config["model"]["gedi_config"]))
        
        print("\n\nComputing descriptors for cads  🛠️ ...")
        compute_descriptors(source_folder=preprocessed_cad_folder,
                            desc_target_folder=cad_descriptors_folder,
                            inds_target_folder=cad_indices_folder,
                            patches_per_pair=config["model"]["gedi_patches_per_pair"],
                            voxel_size=config["model"]["gedi_voxel_size"],
                            model=gedi)
        

def reg_based_identification_pipeline(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
        base_path = config["base"]
        scan_folder = os.path.join(base_path, config["paths"]["scan"]["preprocessed_pcd"])
        cad_folder = os.path.join(base_path, config["paths"]["cad"]["preprocessed_pcd"])
        scan_eigenvalues_folder = os.path.join(base_path, config["paths"]["scan"]["eigenvalues"])
        cad_eigenvalues_folder = os.path.join(base_path, config["paths"]["cad"]["eigenvalues"])
        filtering_coarse_file = os.path.join(base_path, config["paths"]["results"]["filtering_coarse_overall"], config["paths"]["results"]["filtering_coarse_overall_file_name"])
        filtering_coarse_rank_file = os.path.join(base_path, config["paths"]["results"]["filtering_coarse_rank"], config["paths"]["results"]["filtering_coarse_rank_file_name"])
        fine_filtering_folder = os.path.join(base_path, config["paths"]["results"]["filtering_fine"])
        filtering_fine_file = os.path.join(base_path, config["paths"]["results"]["filtering_fine"], config["paths"]["results"]["filtering_fine_file_name"])
        
        print(f"scan_folder len: {len(os.listdir(scan_folder))}")
        print(f"cad_folder len: {len(os.listdir(cad_folder))}")
        
        # Compute coarse score of every scan with every cad
        coarse_df = compute_coarse_score(
            scan_folder=scan_folder,
            cad_folder=cad_folder,
            scan_eigenvalues_folder=scan_eigenvalues_folder,
            cad_eigenvalues_folder=cad_eigenvalues_folder,
            filtering_coarse_folder=filtering_coarse_file,
        )
        
        # Coarse to fine filtering
        # Can't precompute the fine filtering because it depends on the coarse filtering
        rank_coarse_results = {}
        fine_results = {}
        
        
        
        # Use the coarse score to filter the scans
        for scan in tqdm(list(coarse_df.index), desc="Coarse to fine filtering"):
            rank_coarse_results[scan] = {}
            coarse_scan_results = coarse_df.loc[scan]
            coarse_scan_results = coarse_scan_results.to_dict()
            coarse_scan_results = pd.DataFrame.from_dict(coarse_scan_results).T
            

            # to make test, comment the fine filtering, and add as many metrics as you need
            for method in ["score_sim1"]:
                # Sort and apply the threshold
                filtered_candidates = apply_filtering(coarse_scan_results, method=method, threshold=0.6)
                print(f"Filtered candidates for {scan} with {method}: ¬{filtered_candidates}")
                rank = get_rank_coarse(scan, coarse_df, metric=method)
                rank_coarse_results[scan][method] = rank
                
                # Fine filtering...
                fine_result = fine_filtering(scan, filtered_candidates, config)
                fine_results[scan] = fine_result
            
                
        # log scan experiment ranks
        coarse_rank_df = pd.DataFrame.from_dict(rank_coarse_results, orient="index")
        print("Ranking of scan experiments saved here 💾 ===> ", filtering_coarse_rank_file)
        
        path_checker1(fine_filtering_folder)
        pd.DataFrame.from_dict(fine_results).to_json(filtering_fine_file)
        print("Fine filtering results saved here 💾 ===> ", filtering_fine_file)

        
if __name__ == "__main__":
    #config = "/app/bindmount/gedi_data_2/config.yaml"
    # config = "/app/bindmount/gedi_data_real_scan/config_real_scan.yaml"
    config = "/app/bindmount/gedi_data_real_scan_5x315/config_real_scan_5x315.yaml"
    # precompute_pipeline(config)
    reg_based_identification_pipeline(config)
