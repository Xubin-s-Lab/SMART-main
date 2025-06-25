# SMART: spatial multiomics aggregation with graphical metric learning
This repository contains SMART model and jupyter notebooks essential for reproducing the benchmarking outcomes shown in the paper.
![Figure1](https://github.com/user-attachments/assets/7e58d0e5-7491-4fc2-9114-c21586eec3cf)

## Overview!

Spatial multi-omics provide a comprehensive aspect to decipher the microenvironment and heterogeneity in different spatial domain within tissues. To uniformly analyze the spatial multi-omics data, computational methods are desired to integrate and represent multiple omics with spatial information in uniform space. Here, we present SMART, a deep learning model based on graphical neural network aggregation and metric learning. SMART leverages a modality-independent modular and stacking framework with spatial coordinates and adjusts the aggregation using the constructed triplet relationship. SMART excels in accurately identifying spatial regions of anatomical structures or cell types, compatible with spatial datasets of any type and number of omics layers, while demonstrating exceptional computational efficiency and scalability on large datasets. Additionally, the variant of SMART, SMART-MS, expands its functionality to integrate multi-omics spatial data across multiple tissue sections. 

## Requirements
- python==3.9
- muon==0.1.6
- rpy2==3.5.12
- scanpy==1.10.2
- scikit-learn==1.5.1
- torch==2.4.1
- torch_geometric==2.3.0
- anndata==0.10.8
- matplotlib==3.9.2
- tqdm==4.66.5
- numba==0.60.0
- R==4.3.0

## Datasets
The study contains 6 data set types.
The 10X Visium Human Lymph Node data can be accessed from the Gene Expression Omnibus (GEO) with accession code GSE263617 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617). 
The MISAR-seq mouse brain data can be accessed from National Genomics Data Center with accession number OEP003285 (https://www.biosino.org/node/project/detail/OEP003285). 
The Stereo-CITE-seq mouse thymus data can be accessed from BGI STOmics Cloud (https://cloud.stomics.tech/). 
The spatial CUT&Tag–RNA-seq mouse brain data can be accessed at GEO with accession code GSE205055 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055) or UCSC Cell and Genome Browser (https://brain-spatial-omics.cells.ucsc.edu).
The STARmap and RIBOmap mouse brain data can be accessed from Zenodo (https://zenodo.org/record/8041114) or Single Cell Portal (SCP) (https://singlecell.broadinstitute.org/single_cell/study/SCP1835). 
The SPOTS mouse spleen data can be accessed at GEO with accession code GSE198353 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353).
