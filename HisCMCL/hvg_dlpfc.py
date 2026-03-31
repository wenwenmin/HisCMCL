#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scanpy as sc
import pickle
import pandas as pd
import warnings
import os
import scprep as scp
warnings.filterwarnings("ignore")

def intersect_section_genes(adata_list):
    shared = set.intersection(*[set(adata.var_names) for adata in adata_list])
    return list(shared)

def dlpfc_hvg_selection_and_pooling(adata_list, n_top_genes=1000):
    shared = intersect_section_genes(adata_list)
    hvg_bools = []

    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to shared genes
        adata = adata[:, shared]
        print(adata.shape)
        # Preprocess the data
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        # save hvgs
        hvg = adata.var['highly_variable']
        hvg_bools.append(hvg)

    hvg_union = hvg_bools[0]
    hvg_intersection = hvg_bools[0]
    for i in range(1, len(hvg_bools)):
        print(sum(hvg_union), sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]
        print(sum(hvg_intersection), sum(hvg_bools[i]))
        hvg_intersection = hvg_intersection & hvg_bools[i]

    print("Number of HVGs: ", hvg_union.sum())
    print("Number of HVGs (intersection): ", hvg_intersection.sum())

    with open('dlpfc_hvgs_intersection.pickle', 'wb') as handle:
        pickle.dump(hvg_intersection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('dlpfc_hvgs_union.pickle', 'wb') as handle:
        pickle.dump(hvg_union, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Add all the HVGs
    gene_list_path = "D:/dataset/DLPFC/brain_cut_1000.npy"
    gene_list = list(np.load(gene_list_path, allow_pickle=True))

    hvg_union[gene_list] = True

    filtered_exp_mtxs = []
    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to shared genes
        adata = adata[:, shared]
        filtered_exp_mtxs.append(adata[:, gene_list].X.T.toarray())
    return filtered_exp_mtxs

def dlpfc_pool_gene_list(adata_list, n_top_genes=1000):
    shared = intersect_section_genes(adata_list)
    gene_list_path = "D:/dataset/DLPFC/brain_cut_1000.npy"
    gene_list = list(np.load(gene_list_path, allow_pickle=True))

    filtered_exp_mtxs = []
    for adata in adata_list:
        adata.var_names_make_unique()
        filtered_exp_mtxs.append(adata[:, gene_list].X.T.toarray())
    return filtered_exp_mtxs

if __name__ == "__main__":
    # DLPFC sample names
    names = ['151507', '151508', '151509', '151510', '151669', '151670', 
             '151671', '151672', '151673', '151674', '151675', '151676']

    # Load data
    adata_list = []
    for name in names:
        try:
            adata = sc.read_visium(path=f"D:/dataset/DLPFC/{name}",
                                 count_file=f'{name}_filtered_feature_bc_matrix.h5')
            adata.var_names_make_unique()
            adata_list.append(adata)
            print(f"Successfully loaded {name}")
        except Exception as e:
            print(f"Error loading {name}: {str(e)}")

    # Process data
    filtered_mtx = dlpfc_pool_gene_list(adata_list)

    # Save preprocessed matrices
    preprocessed_mtx = []
    for i, mtx in enumerate(filtered_mtx):
        log_transformed_expression = scp.transform.log(scp.normalize.library_size_normalize(mtx))
        preprocessed_mtx.append(log_transformed_expression)
        
        # Create directory if it doesn't exist
        pathset = f"./data/preprocessed_expression_matrices/DLPFC_data/{names[i]}"
        if not os.path.exists(pathset):
            os.makedirs(pathset)
            
        # Save preprocessed matrix
        np.save(f"./data/preprocessed_expression_matrices/DLPFC_data/{names[i]}/preprocessed_matrix.npy",
                log_transformed_expression)
        print(f"DLPFC_data_preprocessed_mtx[{i}]:", log_transformed_expression.shape)
