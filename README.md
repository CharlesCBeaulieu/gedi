# Coarse-to-Fine CAD Retrieval in a Dynamic Database

This repository contains the implementation of a coarse-to-fine CAD model retrieval pipeline designed for use in a dynamic database environment.

The codebase includes:
- Data preprocessing – Prepare and clean the dataset for processing.
- Feature precomputation – Extract and store descriptors or features for faster retrieval.
- Coarse filtering – Perform an initial filtering step to quickly reduce the search space.
- Fine filtering – Apply precise registration and similarity measures to refine matches.
- Result saving – Store retrieval results for further evaluation and analysis.

Evaluation Notebooks

Several Jupyter notebooks are provided for evaluating different parts of the pipeline:
- eval_coarse.ipynb – Evaluate the coarse filtering stage.
- eval_fine.ipynb – Evaluate the fine filtering stage.
- evaluate_data.ipynb – Perform a global assessment of the dataset and pipeline performance.

