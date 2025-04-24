import yaml
from reg_based_identification_utils import *


def precompute_pipeline(config_path, Preprocessing=True, Eigenvalues=True, Descriptors=True):
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
        
        if Preprocessing:
            print("\n\nPreprocessing scans 🛠️ ...")
            preprocess_scan(source_folder=raw_scan_folder,
                            target_folder=preprocessed_scan_folder,
                            patches_per_pair=config["model"]["gedi_patches_per_pair"])
            
            print("\n\nPreprocessing cads  🛠️ ...")
            preprocess_CAD(source_folder=raw_cad_folder,
                        target_folder=preprocessed_cad_folder,
                        patches_per_pair=config["model"]["gedi_patches_per_pair"],
                        scale_factor=config["preprocess_params"]["cad"]["scale_factor"])
        
        if Eigenvalues:
            print("\n\nComputing eigenvalues for scans 🛠️ ... ")
            compute_eigenvalues(source_folder=preprocessed_scan_folder,
                                target_folder=scan_eigenvalues_folder)
            
            print("\n\nComputing eigenvalues for cads 🛠️ ... ")
            compute_eigenvalues(source_folder=preprocessed_cad_folder,
                                target_folder=cad_eigenvalues_folder)
        
        if Descriptors:
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


def reg_based_identification_pipeline(config_path, fine_filter=True):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
        base_path = config["base"]
        scan_folder = os.path.join(base_path, config["paths"]["scan"]["preprocessed_pcd"])
        cad_folder = os.path.join(base_path, config["paths"]["cad"]["preprocessed_pcd"])
        scan_eigenvalues_folder = os.path.join(base_path, config["paths"]["scan"]["eigenvalues"])
        cad_eigenvalues_folder = os.path.join(base_path, config["paths"]["cad"]["eigenvalues"])
        filtering_coarse_folder = os.path.join(base_path, config["paths"]["results"]["filtering_coarse_overall"])
        filtering_coarse_file = os.path.join(base_path, config["paths"]["results"]["filtering_coarse_overall"], config["paths"]["results"]["filtering_coarse_overall_file_name"])
        filtering_coarse_rank_file = os.path.join(base_path, config["paths"]["results"]["filtering_coarse_rank"], config["paths"]["results"]["filtering_coarse_rank_file_name"])
        filtering_coarse_overall_file = os.path.join(base_path, config["paths"]["results"]["filtering_coarse_overall"], config["paths"]["results"]["filtering_coarse_overall_file_name"])
        fine_filtering_folder = os.path.join(base_path, config["paths"]["results"]["filtering_fine"])
        filtering_fine_file = os.path.join(base_path, config["paths"]["results"]["filtering_fine"], config["paths"]["results"]["filtering_fine_file_name"])
        path_checker1(filtering_coarse_folder)
        
        try:
            print(f"scan_folder len: {len(os.listdir(scan_folder))}")
            print(f"cad_folder len: {len(os.listdir(cad_folder))}")
        except FileNotFoundError as e:
            print("Check for Precomputing... preprocess, eigenvalues, descriptors")
        
        # Compute coarse score of every scan with every cad
        coarse_df = compute_coarse_score(
            scan_folder=scan_folder,
            cad_folder=cad_folder,
            scan_eigenvalues_folder=scan_eigenvalues_folder,
            cad_eigenvalues_folder=cad_eigenvalues_folder,
            filtering_coarse_folder=filtering_coarse_file,
        )
        print(coarse_df)
        
        # Coarse to fine filtering
        # Can't precompute the fine filtering because it depends on the coarse filtering
        rank_coarse_results = {}
        fine_rank_results = {}
        fine_results = {}
        
        # Use the coarse score to filter the scans
        for scan in tqdm(list(coarse_df.index), desc="Coarse to fine filtering"):
            rank_coarse_results[scan] = {}
            coarse_scan_results = coarse_df.loc[scan]
            coarse_scan_results = coarse_scan_results.to_dict()
            coarse_scan_results = pd.DataFrame.from_dict(coarse_scan_results).T
            

            # to make test, comment the fine filtering, and add as many metrics as you need
            for method in ["score_sim1", "score_sim2"]:
                # Sort and apply the threshold + get coarse rank 
                filtered_candidates= apply_filtering(coarse_scan_results, method="score_sim1", threshold=0.04618)
                print("Scan : ", scan)
                print("Metric : ", method)
                print("Filtered candidates len: ", len(filtered_candidates))
                coarse_rank = get_rank_coarse(scan, coarse_df, metric=method)
                rank_coarse_results[scan][method] = coarse_rank
                print("\n")
                
                if fine_filter:
                    #Fine filtering...
                    fine_result = fine_filtering(scan, filtered_candidates, config)                
                    fine_rank = get_rank_fine(scan, fine_result)
                    fine_rank_results[scan] = fine_rank
                    fine_results[scan] = fine_result
                    
                
            
        # for key, val in fine_rank_results.items():
        #     print(f"Scan {key} has a rank of {val} candidates")
    
        coarse_df.to_json(filtering_coarse_overall_file)
        print("Coarse filtering results saved here 💾 ===> ", filtering_coarse_overall_file)
        
        # log scan experiment ranks
        pd.DataFrame.from_dict(rank_coarse_results, orient="index").to_json(filtering_coarse_rank_file)
        print("Ranking of scan experiments saved here 💾 ===> ", filtering_coarse_rank_file)
        
        if fine_filter:
            path_checker1(fine_filtering_folder)
            pd.DataFrame.from_dict(fine_results).to_json(filtering_fine_file)
            print("Fine filtering results saved here 💾 ===> ", filtering_fine_file)



# def test_coarse_filtering(config_path):
#     """
#     Test the coarse filtering alone
#     this permit also to fix the filtering value
#     """
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)
        
#         base_path = config["base"]
#         scan_folder = os.path.join(base_path, config["paths"]["scan"]["preprocessed_pcd"])
#         cad_folder = os.path.join(base_path, config["paths"]["cad"]["preprocessed_pcd"])
#         scan_eigenvalues_folder = os.path.join(base_path, config["paths"]["scan"]["eigenvalues"])
#         cad_eigenvalues_folder = os.path.join(base_path, config["paths"]["cad"]["eigenvalues"])
#         filtering_coarse_file = os.path.join(base_path, config["paths"]["results"]["filtering_coarse_overall"], config["paths"]["results"]["filtering_coarse_overall_file_name"])
#         filtering_coarse_folder = os.path.join(base_path, config["paths"]["results"]["filtering_coarse_overall"])
#         filtering_coarse_rank_file = os.path.join(base_path, config["paths"]["results"]["filtering_coarse_rank"], config["paths"]["results"]["filtering_coarse_rank_file_name"])
        
#         # Compute coarse score of every scan with every cad
#         coarse_df = compute_coarse_score(
#             scan_folder=scan_folder,
#             cad_folder=cad_folder,
#             scan_eigenvalues_folder=scan_eigenvalues_folder,
#             cad_eigenvalues_folder=cad_eigenvalues_folder,
#             filtering_coarse_folder=filtering_coarse_folder,
#         )
        
#         rank_coarse_results = {}
#         fail = 0
#         coarse_rank = []
        
#         # Use the coarse score to filter the scans
#         for scan in tqdm(list(coarse_df.index), desc="Coarse to fine filtering"):
#             rank_coarse_results[scan] = {}
#             coarse_scan_results = coarse_df.loc[scan]
#             coarse_scan_results = coarse_scan_results.to_dict()
#             coarse_scan_results = pd.DataFrame.from_dict(coarse_scan_results).T
            
#             # to make test, comment the fine filtering, and add as many metrics as you need
#             for method in ["score_sim1"]:
#                 # Sort and apply the threshold
#                 Ktop = 30
#                 filtered_candidates = apply_filtering(scan, coarse_scan_results, method=method, Ktop=Ktop, threshold=False)
                
#                 rank = get_rank_coarse(scan, coarse_df, metric=method)
#                 print(f"Rank of {scan} : {rank}")
#                 rank_coarse_results[scan][method] = rank
                
#                 if rank > Ktop:
#                     fail += 1
#                     print(f"Fail for {scan} with rank {rank}")
                
#         print(f"Fail: {fail} out of {len(coarse_df)}")
        
#         # save the coarse_df to a file
#         pd.DataFrame.from_dict(rank_coarse_results, orient="index").to_json(filtering_coarse_file)
#         print("Ranking of scan experiments saved here 💾 ===> ", filtering_coarse_file)
            


def main():
    # configs = [
    #     "/app/bindmount/gedi_data_2/config.yaml",
    #     "/app/bindmount/gedi_data_real_scan/config_real_scan.yaml",
    #     "/app/bindmount/gedi_data_real_scan_5x315/config_real_scan_5x315.yaml",
    #     "/app/bindmount/gedi_data_4meters_scans/config_4meters.yaml",
    # ]
    
    # configs_4m = [
    #     "/app/bindmount/gedi_data_subset25x25_4m/config_subset25x25_4m.yaml",
    #     "/app/bindmount/gedi_data_subset25x50_4m/config_subset25x50_4m.yaml",
    #     "/app/bindmount/gedi_data_subset25x100_4m/config_subset25x100_4m.yaml",
    # ]
    
    config_new = ["/app/bindmount/test_set_30x30/config_30x30.yaml",
                   "/app/bindmount/test_set_30x50/config_30x50.yaml",
                   "/app/bindmount/test_set_30x100/config_30x100.yaml"]
    
    for config in config_new:
        print(f"Running pipeline for config: {config}")
        #precompute_pipeline(config_path=config, Preprocessing=True, Eigenvalues=True, Descriptors=True)
        reg_based_identification_pipeline(config_path=config, fine_filter=True)


if __name__ == "__main__":
    main()
