## CLL_MRD

## Code repository for manuscript "Automated CLL cell cluster detection using a weakly supervised approach and CLL MRD flow cytometry data"

![diagram1](https://github.com/user-attachments/assets/1a834b9b-0446-4c57-a637-e6fbc9392fa8)

Wikum Dinalankara, Luigi Marchionni, Paul D. Simonson

Abstract

Minimal residual disease detection is routinely performed as part of post-diagnostic treatment plans for many types of cancer, for which multiparameter flow cytometry is one possible modality frequently used in the clinic. We propose a machine learning approach for binary prediction of minimal residual disease status with flow cytometry data. Our method involves the projection of cells from the original feature space to a low dimensional embedding in which cells are clustered by similarity and the cluster wise cell proportions are used for prediction as well as regression. We demonstrate the applicability of our method with respect to a cohort of chronic lymphocytic leukemia to obtain high levels of accuracy and contrast our approach with other proposed machine learning methods towards minimal residual disease prediction.

Notes for running:

(1) For each python script, set the DIRPATH variable to the folder corresonding to the fcs files and annotation table.

(2) Run 2_UMAP.py first to create the UMAP and the case projections (set the number of cells to be sample if necessary, default = 1000000). This will create a folder in ./obj with a RUN ID.

(3) Run 3_cluster.py after setting the RUN ID from (2) inside the script to perform classification and regression corresponding to the UMAP projections obtained. Set the number of clusters to be estimated by k-means if necessary (default=1000).

