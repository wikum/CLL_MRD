## CLL_MRD

## Code repository for manuscript "Automated CLL cell cluster detection using a weakly supervised approach and CLL MRD flow cytometry data"

![diagram1](https://github.com/user-attachments/assets/1a834b9b-0446-4c57-a637-e6fbc9392fa8)

Wikum Dinalankara, Chandler Sy, Jiani Chai, Paul B. Barone, Luigi Marchionni, and Paul D. Simonson

Abstract

Minimal residual disease detection is routinely performed as part of post-diagnostic treatment plans for many types of cancer, for which multiparameter flow cytometry is one possible modality frequently used in the clinic. We propose a machine learning approach for binary prediction of minimal residual disease status with flow cytometry data. Our method involves the projection of cells from the original feature space to a low dimensional embedding in which cells are clustered by similarity and the cluster wise cell proportions are used for prediction as well as regression. We demonstrate the applicability of our method with respect to a cohort of chronic lymphocytic leukemia to obtain high levels of accuracy and contrast our approach with other proposed machine learning methods towards minimal residual disease prediction.

Notes for running:

(1) Set the DIRPATH variable in settings.ini to the folder corresonding to the fcs files and annotation table. Set K to be the number of folds to be generated for cross validation and M to the number of cells to sample.

(2) Run 2_UMAP.py first to create the UMAP and the case projections (set the number of cells to be sample if necessary through settings.ini or manually in code; default = 1000000). This will create a folder in ./obj with a RUN ID.

(3) Run 3_projections.py after setting the RUN ID from (2) inside the script to generate UMAP projections for each fold.

(4) Run 4_cluster.py to run clustering for each fold after setting the RUN ID from (2). This will generate cluster labels for each cell and aggregate them into a numpy matrix. 

(5) Perform classification, regression, and clusters of interest discovery. (Note: this section of the code will be completed upon publication.)

