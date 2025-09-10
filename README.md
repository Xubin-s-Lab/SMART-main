# SMART: Spatial multi-omic aggregation using graph neural networks and metric learning
This repository contains SMART model and jupyter notebooks essential for reproducing the benchmarking outcomes shown in the paper.
![图片1](https://github.com/user-attachments/assets/2d998716-1917-4c7e-b75f-a66ff46828c2)



## 1. Overview
Spatial multi-omics enables the exploration of tissue microenvironments and heterogeneity from the perspective of different omics modalities across distinct spatial domains within tissues. To jointly analyze the spatial multi-omics data, computational methods are desired to integrate multiple omics with spatial information into a unified space. Here, we present SMART (Spatial Multi-omic Aggregation using gRaph neural networks and meTric learning), a computational framework for spatial multi-omic integration. SMART leverages a modality-independent modular and stacking framework with spatial coordinates and adjusts the aggregation using triplet relationships. SMART excels at accurately identifying spatial regions of anatomical structures, compatible with spatial datasets of any type and number of omics layers, while demonstrating exceptional computational efficiency and scalability on large datasets. Moreover, a variant of SMART, SMART-MS, expands its capabilities to integrate spatial multi-omics data across multiple tissue sections. In summary, SMART provides a versatile, efficient, and scalable solution for integrating spatial multi-omics data.

## 2. Environment setup and code  compilation

### 2.1 Download the package

The package can be download by running the following command in the terminal:

```bash
git clone https://github.com/ws6tg/SMART-main.git
```

Then, use 

```
cd SMART
```

to access the downloaded folder.

If the "git clone" command does not work with your system, you can also download the zip file from the website https://github.com/ws6tg/SMART-main.git and decompress it. Then, the folder that you need to access is SMART-main.

### 2.2 Environment setup

The package has been successfully tested in a Linux environment of python version 3.9.23, R==4.3.0,and so on. An option to set up the environment is to use Conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

You can use the following command to create an environment for SMART:

```
# 1. create conda environment
conda create -n smart python=3.9.23 -y
# 2. activate conda environment
conda activate smart
# 3. use conda install R(r-base=4.3.0)
conda install -c conda-forge r-base=4.3.0 -y
# 4. use pip install other dependencies (root in '\SMART-main')
pip install -r requirements.txt
```

You need to install `mclust` package by following these steps:

```
# 1. Activate the conda environment
conda activate smart
# 2. Start the R console
R
# 3. Install the mclust package
install.packages("mclust", repos = "https://cloud.r-project.org")
# 4. Exit the R console
q()
```

Please install Jupyter Notebook from https://jupyter.org/install. For example, you can run

```
pip install notebook
```

in the terminal to install the classic Jupyter Notebook.

### 2.3 Use "pip install" to install the SMART package

Please run 

```
pip install bio-SMART
```

in the terminal.

## 3. Tutorials

The step-by-step guides for closely replicating the SMART results on XXX are accessible at: [Tutorials]() and [SMART Tutorials on Read the Docs](). Furthermore, all the processed data required to reproduce the figures presented in the manuscript can be found at Zenodo under the DOI: .

## 4. Datasets

The 10X Visium Human Lymph Node data can be accessed from the Gene Expression Omnibus (GEO) with accession code GSE263617 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617). The MISAR-seq mouse brain data can be accessed from National Genomics Data Center with accession number OEP003285 (https://www.biosino.org/node/project/detail/OEP003285). The Stereo-CITE-seq data can be accessed from BGI STOmics Cloud (https://cloud.stomics.tech/). The spatial CUT&Tag–RNA-seq and spatial ATAC–RNA-seq mouse brain data can be accessed at GEO with accession code GSE205055 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055) or UCSC Cell and Genome Browser (https://brain-spatial-omics.cells.ucsc.edu). The STARmap and RIBOmap mouse brain data can be accessed from Zenodo (https://zenodo.org/record/8041114) or Single Cell Portal (SCP) (https://singlecell.broadinstitute.org/single_cell/study/SCP1835). The SPOTS mouse spleen data can be accessed at GEO with accession code GSE198353 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353). The 10X Visium Human Tonsil data can be accessed from https://zenodo.org/records/12654113/preview/data_imputation.zip?include_deleted=0#tree_item0.
