import yaml
from reg_based_identification_utils import *


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
        print("\n\nPreprocessing Scans and CADs 🛠️ ... ")
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

        # Precomputing descriptors
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


def identification_based_pipeline():
    pass


def main():
    precompute_pipeline(config_path="/app/bindmount/gedi_data/config.yaml")


if __name__ == "__main__":
    main()
