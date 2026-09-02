# SMART: Spatial multi-omic aggregation using graph neural networks and metric learning
This repository contains SMART model and jupyter notebooks essential for reproducing the benchmarking outcomes shown in the paper.
![图片1](https://github.com/user-attachments/assets/2d998716-1917-4c7e-b75f-a66ff46828c2)



## 1. Overview
Spatial multi-omics enables the exploration of tissue microenvironments and heterogeneity from the perspective of different omics modalities across distinct spatial domains within tissues. To jointly analyze the spatial multi-omics data, computational methods are desired to integrate multiple omics with spatial information into a unified space. Here, we present SMART (Spatial Multi-omic Aggregation using gRaph neural networks and meTric learning), a computational framework for spatial multi-omic integration. SMART leverages a modality-independent modular and stacking framework with spatial coordinates and adjusts the aggregation using triplet relationships. SMART excels at accurately identifying spatial regions of anatomical structures, compatible with spatial datasets of any type and number of omics layers, while demonstrating exceptional computational efficiency and scalability on large datasets. Moreover, a variant of SMART, SMART-MS, expands its capabilities to integrate spatial multi-omics data across multiple tissue sections. In summary, SMART provides a versatile, efficient, and scalable solution for integrating spatial multi-omics data.

## 2. Environment setup and code  compilation

Note: Installation, including dependencies, typically completes within 10–15 minutes on a normal desktop computer.

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

The SMART package has been successfully tested in a **Linux environment** with the following key components:

- **Python**: 3.9.23
- **R**: 4.3.0
- **PyTorch** and its companion packages: `torch-scatter`, `torch-sparse`, `torch-geometric`.
- Other dependencies listed in `requirements.txt`

It is recommended to use **Conda** to manage the environment. Conda simplifies the installation of Python, R, and other libraries. See the Conda official documentation(https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for installation instructions.

You can use the following command to create an environment for SMART:

```
# 1. create conda environment
conda create -n smart python=3.9.23 -y
# 2. activate conda environment
conda activate smart
# 3. install R (R base 4.3.0) from conda-forge
conda install -c conda-forge r-base=4.3.0 -y
# 4. install Python dependencies from requirements.txt (run from root directory)
pip install -r requirements.txt
```

#### Installing PyTorch Geometric companion packages

PyTorch Geometric (PyG) requires additional packages (`torch-scatter`, `torch-sparse`). These packages need to be installed from PyG’s wheel files (https://data.pyg.org/whl/) compatible with your PyTorch and CUDA versions. For example:

```
# Example for PyTorch 2.4.1 + CUDA 12.1
pip install torch-scatter-2.1.2+pt24cu121-cp39-cp39-linux_x86_64.whl
pip install torch-sparse-0.6.18+pt24cu121-cp39-cp39-linux_x86_64.whl
```

See the official installation guide for the correct wheels for your platform: PyTorch Geometric installation (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

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

The step-by-step guides for closely replicating the SMART results on **Simulated multi-omics data, 10x human lymph node data, MISAR-seq mouse brain data, P22 mouse brain section data and 10x human tonsil multi slice data** are accessible at: [Tutorials](https://github.com/ws6tg/SMART-main/tree/main/tutorials) and [SMART Tutorials on Read the Docs](https://smart-tutorials.readthedocs.io/). Furthermore, all the processed data required to reproduce the figures presented in the manuscript can be found at Zenodo under the DOI: https://doi.org/10.5281/zenodo.17093158.

## 4. Version notes

- **v0.1.2** ([GitHub release](https://github.com/Xubin-s-Lab/SMART-main/releases/tag/0.1.2), also on PyPI as `bio-SMART==0.1.2`): the version used to generate all five tutorials above. In this version, `SMART.encoders` and `SMART.decoders` are stored in plain Python lists, so their parameters are not registered with the optimizer: during training, only the fusion layer (`fc`) is updated, while the graph encoder/decoder weights keep their initial values. To exactly reproduce the recorded tutorial outputs, please use v0.1.2.

- **v0.1.3** ([GitHub release](https://github.com/Xubin-s-Lab/SMART-main/releases/tag/0.1.3), current): `SMART.encoders` and `SMART.decoders` are now wrapped in `torch.nn.ModuleList`, so all encoder/decoder parameters are registered and trained. Because the set of trainable parameters changed, training dynamics differ from v0.1.2: with identical hyperparameters and random seeds, v0.1.3 produces different results from the tutorials' recorded outputs, and per-dataset learning-rate / early-stopping settings may benefit from recalibration. The current `main` branch additionally includes a compatibility improvement for the mclust (rpy2) call.

## 5. Raw datasets

The 10X Visium Human Lymph Node data can be accessed from the Gene Expression Omnibus (GEO) with accession code GSE263617 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617). The MISAR-seq mouse brain data can be accessed from National Genomics Data Center with accession number OEP003285 (https://www.biosino.org/node/project/detail/OEP003285). The Stereo-CITE-seq data can be accessed from BGI STOmics Cloud (https://cloud.stomics.tech/). The spatial CUT&Tag–RNA-seq and spatial ATAC–RNA-seq mouse brain data can be accessed at GEO with accession code GSE205055 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055) or UCSC Cell and Genome Browser (https://brain-spatial-omics.cells.ucsc.edu). The STARmap and RIBOmap mouse brain data can be accessed from Zenodo (https://zenodo.org/record/8041114) or Single Cell Portal (SCP) (https://singlecell.broadinstitute.org/single_cell/study/SCP1835). The SPOTS mouse spleen data can be accessed at GEO with accession code GSE198353 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353). The 10X Visium Human Tonsil data can be accessed from https://zenodo.org/records/12654113/preview/data_imputation.zip?include_deleted=0#tree_item0.

## 6. Reproducibility

To reproduce the results of SMART and the compared methods, please use the code provided in the **SMART-reproduce** branch and the **benchmarks** folder in the main branch.  


## 7. Contact information

Please contact us if you have any questions:

- Qiyi Chen (chenqiyi2022@email.szu.edu.cn);
- Weiliang Huang (wlhuang32@gmail.com);
- Xubin Zheng (xbzheng@gbu.edu.cn).

## 8. Copyright information

Please see the "LICENSE" file for the copyright information.