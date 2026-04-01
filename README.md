# HisCMCL: Cross-Modal Contrastive Learning with Hierarchical Multi-Scale Fusion for Spatial Expression Prediction
## Overview
HisCMCL is an innovative deep learning framework developed to improve the accuracy of spatial gene expression prediction. This framework combines features from Hematoxylin and Eosin (H&E) stained tissue images with spatial gene expression data, leveraging contrastive learning techniques for accurate gene expression predictions. By integrating local and global image features, HisCMCL addresses the challenges of extracting spatially accurate gene expression data from histological images, offering a promising solution to the limitations of traditional spatial transcriptomics (ST) methods.

One of the key innovations of HisCMCL is its multi-level feature fusion strategy. Unlike conventional methods that focus on either local or global features, HisCMCL integrates image features at multiple scales—spot-level, neighborhood-level, and global-level. This hierarchical fusion enables the model to capture both fine-grained and broad contextual information, significantly enhancing prediction accuracy. The framework also employs contrastive learning to maximize the alignment between image features and gene expression data. By optimizing the similarity of correctly paired images and gene expression data while minimizing the similarity of mismatched pairs, HisCMCL ensures better feature alignment and improves the overall predictive performance of the model.

Experimental results have shown that HisCMCL outperforms existing methods in multiple datasets, demonstrating its superiority in accurately identifying cancer-related genes, immune markers, and spatial regions annotated by pathologists. In addition to quantitative improvements, qualitative assessments of the model’s predictions reveal that HisCMCL more effectively preserves the underlying gene expression patterns, providing deeper insights into spatial gene expression. This capability is particularly valuable in fields like cancer research, where understanding the spatial organization of gene expression is crucial for discovering disease mechanisms.

Furthermore, HisCMCL bridges the gap between histological images and high-resolution gene expression data, offering a cost-effective alternative to high-resolution spatial transcriptomics techniques. It has the potential to revolutionize spatial gene expression analysis by making it more accessible and efficient, with strong applications in clinical settings for disease diagnosis and treatment development.
![overview.png](overview.png)

## System environment
Required package:
- PyTorch >= 2.1.0
- scanpy >= 1.8
- python >=3.9

## Datasets
The publicly available ST datasets were used in this study. You can download them from https://zenodo.org/records/18030446 or find them on the following websites：
-  human HER2-positive breast tumor ST data from https://github.com/almaan/her2st/.
-  human cutaneous squamous cell carcinoma 10x Visium data from GSE144240.
-  10x Genomics Visium data and Swarbrick’s Laboratory Visium data from https://doi.org/10.48610/4fb74a9.

## Repository Structure

```text
HisCMCL/
├── requirements/           # Environment Configuration
├── HisCMCL/            # Core modules for model and utilities
├── Baseline/         # Baseline methods for comparison
├── Tutorial/         # Jupyter notebooks for tutorials
└── README.md         # Documentation
```

## HisCMCL pipeline

- Run `hvg.py` generation of highly variable genes.
- Run `train.py`
- Run `evel.py`

## Tutorial
This tutorial notebook provides a small demo for the Alex dataset, including:
- Screening of highly variable genes
- Evaluation of gene expression prediction
- Downstream analysis on a subset of the dataset

Run the tutorial script to process your data step by step:

```bash
jupyter notebook tutorial.ipynb
```
## Contact details

If you have any questions, please contact liuchengju@stu.ynu.edu.cn and minwenwen@yun.edu.cn.
