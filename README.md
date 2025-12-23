# HisCMCL
## Overview
HisCMCL is an innovative deep learning framework designed to enhance the accuracy of spatial gene expression prediction. This framework combines tissue image features stained with Hematoxylin and Eosin (H&E) and spatial gene expression data, using contrastive learning methods to make predictions.

Unlike traditional methods, HisCMCL employs a multi-level feature fusion strategy, integrating local, neighbor, and global image features to capture more layers of information, thereby improving the accuracy of gene expression predictions. The contrastive learning mechanism within the framework maximizes the similarity between correctly matched image and gene expression data while minimizing the similarity of mismatched data, ensuring better alignment of image and gene expression features, thus enhancing the model's performance.

Experimental results demonstrate that HisCMCL outperforms existing prediction methods across multiple datasets, particularly excelling in identifying cancer-related genes, immune markers, and spatial regions annotated by pathologists. Moreover, by incorporating spatial information, HisCMCL not only accurately predicts gene expression patterns but also uncovers complex biological features, providing strong support for cancer research and other fields.

![overview.jpg](overview.jpg)

## System environment
Required package:
- PyTorch >= 2.1.0
- scanpy >= 1.8
- python >=3.9

## Datasets
The publicly available ST datasets were used in this study. You can download them from https://zenodo.org/records/13117873 or find them on the following websites：
-  human HER2-positive breast tumor ST data from https://github.com/almaan/her2st/.
-  human cutaneous squamous cell carcinoma 10x Visium data from GSE144240.
-  10x Genomics Visium data and Swarbrick’s Laboratory Visium data from https://doi.org/10.48610/4fb74a9.

## HisCMCL pipeline

- Run `hvg_her2st.py` generation of highly variable genes.
- Run `train.py`
- Run `evel.py`

