import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from model import mclSTExp_Attention
from dataset import SKIN, HERDataset, DATA_BRAIN
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import pickle
from utils import get_R
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 确保必要的目录存在
def ensure_dirs():
    dirs = [
        "./model_result/Brain",
        "./embedding_result/Brain",
        "./data/preprocessed_expression_matrices/DLPFC_data"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def build_loaders_inference(dataset):
    datasets = []
    if dataset == "her2st":
        for i in range(32):
            dataset = HERDataset(train=False, fold=i)
            print(dataset.id2name[0])
            datasets.append(dataset)
    if dataset == "cSCC":
        for i in range(9):
            dataset = SKIN(train=False, fold=i)
            print(dataset.id2name[0])
            datasets.append(dataset)
    if dataset == "Brain":
        for i in range(12):
            dataset = DATA_BRAIN(train=False, fold=i)
            print(dataset.id2name[0])
            datasets.append(dataset)

    dataset = torch.utils.data.ConcatDataset(datasets)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Finished building loaders")
    return test_loader


def get_image_embeddings(model_path, model, dataset):
    test_loader = build_loaders_inference(dataset)

    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')
        new_key = new_key.replace('well', 'spot')
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    test_image_embeddings = []
    spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)

            spot_feature = batch["expression"].cuda()
            x = batch["position"][:, 0].long().cuda()
            y = batch["position"][:, 1].long().cuda()
            centers_x = model.x_embed(x)
            centers_y = model.y_embed(y)
            spot_embeddings.append(model.spot_projection(spot_feature + centers_x + centers_y))
    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings)


def find(spot_embeddings, query_embeddings, top_k=1):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)

    return indices.cpu().numpy()


def save_embeddings(model_path, save_path, datasize, dim, dataset):
    os.makedirs(save_path, exist_ok=True)

    model = mclSTExp_Attention(
        encoder_name="densenet121",
        temperature=1.,
        image_dim=1024,
        spot_dim=dim,
        projection_dim=256,
        heads_num=16,
        heads_dim=128,
        head_layers=2,
        dropout=0.,
        use_multiscale=False
    ).cuda()

    img_embeddings_all, spot_embeddings_all = get_image_embeddings(model_path, model, dataset)

    img_embeddings_all = img_embeddings_all.cpu().numpy()
    spot_embeddings_all = spot_embeddings_all.cpu().numpy()
    print("img_embeddings_all.shape", img_embeddings_all.shape)
    print("spot_embeddings_all.shape", spot_embeddings_all.shape)

    for i in range(len(datasize)):
        index_start = sum(datasize[:i])
        index_end = sum(datasize[:i + 1])
        image_embeddings = img_embeddings_all[index_start:index_end]
        spot_embeddings = spot_embeddings_all[index_start:index_end]
        print("image_embeddings.shape", image_embeddings.shape)
        print("spot_embeddings.shape", spot_embeddings.shape)
        np.save(save_path + "img_embeddings_" + str(i + 1) + ".npy", image_embeddings.T)
        np.save(save_path + "spot_embeddings_" + str(i + 1) + ".npy", spot_embeddings.T)


def get_embedding(SAVE_EMBEDDINGS, dataset):
    if SAVE_EMBEDDINGS:
        for fold in range(12):
            save_embeddings(model_path=f"./model_result/{dataset}/{names[fold]}/best_{fold}.pt",
                            save_path=f"./embedding_result/{dataset}/embeddings_{fold}/",
                            datasize=datasize, dim=84, dataset=dataset)  # 171


def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)

    mse = np.mean((y_true - y_pred) ** 2)

    mae = np.mean(np.abs(y_true - y_pred))

    correlations = []
    for i in range(y_true.shape[1]):
        corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
        correlations.append(corr)
    mean_corr = np.mean(correlations)

    return {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'mean_correlation': mean_corr,
        'correlations': correlations
    }

def plot_metrics(metrics, save_dir, fold, name):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(metrics['correlations'], bins=50)
    plt.title('Distribution of Gene-wise Correlations')
    plt.xlabel('Correlation')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'correlation_distribution_{fold}_{name}.png'))
    plt.close()

    with open(os.path.join(save_dir, f'metrics_{fold}_{name}.txt'), 'w') as f:
        f.write(f"R2 Score: {metrics['r2']:.4f}\n")
        f.write(f"MSE: {metrics['mse']:.4f}\n")
        f.write(f"MAE: {metrics['mae']:.4f}\n")
        f.write(f"Mean Correlation: {metrics['mean_correlation']:.4f}\n")


if __name__ == "__main__":
    SAVE_EMBEDDINGS = True
    dataset = "Brain"
    names = []
    if dataset == "her2st":
        slice_num = 32
        names = os.listdir(r".\dataset\Her2st\data/ST-cnts")
        names.sort()
        names = [i[:2] for i in names][1:33]
    if dataset == "cSCC":
        slice_num = 12
        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)
    if dataset == "Brain":
        slice_num = 12
        names = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673',
                 '151674', '151675', '151676']

    ensure_dirs()

    datasize = []
    for name in names:
        data_path = f"./data/preprocessed_expression_matrices/DLPFC_data/{name}/preprocessed_matrix.npy"
        if not os.path.exists(data_path):
            exit(1)
        datasize.append(np.load(data_path).shape[1])

    for fold in range(9, 10):  # 12
        model_path = f"./model_result/{dataset}/{names[fold]}/best_{fold}.pt"
        if not os.path.exists(model_path):
            exit(1)

    get_embedding(SAVE_EMBEDDINGS, dataset)

    spot_expressions = [np.load(f"./data/preprocessed_expression_matrices/DLPFC_data/{name}/preprocessed_matrix.npy")
                        for name in names]
    hvg_pcc_list = []
    heg_pcc_list = []
    mse_list = []
    mae_list = []
    r2_list = []

    for fold in range(slice_num):
        print(f"evaluating: {fold} slice name: {names[fold]} ")

        save_path = f"./embedding_result/{dataset}/embeddings_{fold}/"
        if not os.path.exists(save_path):
            continue

        try:
            spot_embeddings = [np.load(save_path + f"spot_embeddings_{i + 1}.npy") for i in range(12)]
            image_embeddings = np.load(save_path + f"img_embeddings_{fold + 1}.npy")
        except FileNotFoundError as e:
            continue

        image_query = image_embeddings
        expression_gt = spot_expressions[fold]
        spot_embeddings = spot_embeddings[:fold] + spot_embeddings[fold + 1:]
        spot_expressions_rest = spot_expressions[:fold] + spot_expressions[fold + 1:]

        spot_key = np.concatenate(spot_embeddings, axis=1)
        expression_key = np.concatenate(spot_expressions_rest, axis=1)

        os.makedirs(save_path, exist_ok=True)
        if image_query.shape[1] != 256:
            image_query = image_query.T
        if expression_gt.shape[0] != image_query.shape[0]:
            expression_gt = expression_gt.T
        if spot_key.shape[1] != 256:
            spot_key = spot_key.T
        if expression_key.shape[0] != spot_key.shape[0]:
            expression_key = expression_key.T

        indices = find(spot_key, image_query, top_k=800)
        spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
        spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))

        for i in range(indices.shape[0]):
            query_vec = image_query[i:i+1, :]
            key_vecs = spot_key[indices[i, :], :]

            correlations = []
            for j in range(key_vecs.shape[0]):
                corr, _ = pearsonr(query_vec.flatten(), key_vecs[j])
                correlations.append(corr)
            correlations = np.array(correlations)

            weights = (correlations + 1) / 2
            weights = weights / np.sum(weights)
            
            spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0, weights=weights)
            spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0, weights=weights)

        true = expression_gt
        pred = spot_expression_pred

        true = np.log1p(true)
        pred = np.log1p(pred)

        gene_list_path = "D:\dataset\DLPFC/brain_cut_1000.npy"
        gene_list = list(np.load(gene_list_path, allow_pickle=True))
        adata_ture = anndata.AnnData(true)
        adata_pred = anndata.AnnData(pred)

        adata_pred.var_names = gene_list
        adata_ture.var_names = gene_list

        gene_correlations = []
        for gene_idx in range(adata_ture.X.shape[1]):
            corr, _ = pearsonr(adata_ture.X[:, gene_idx], adata_pred.X[:, gene_idx])
            gene_correlations.append(corr)
        gene_correlations = np.array(gene_correlations)

        top_50_genes_indices = np.argsort(gene_correlations)[::-1][:50]
        top_50_genes_names = adata_ture.var_names[top_50_genes_indices]
        top_50_genes_expression = adata_ture[:, top_50_genes_names]
        top_50_genes_pred = adata_pred[:, top_50_genes_names]

        metrics = calculate_metrics(adata_ture.X, adata_pred.X)
        mse_list.append(metrics['mse'])
        mae_list.append(metrics['mae'])
        r2_list.append(metrics['r2'])

        plot_metrics(metrics, save_path, fold, names[fold])

        print(f"R2 Score: {metrics['r2']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Mean Correlation: {metrics['mean_correlation']:.4f}")

        heg_pcc, heg_p = get_R(top_50_genes_pred, top_50_genes_expression)
        hvg_pcc, hvg_p = get_R(adata_pred, adata_ture)
        hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]

        heg_pcc_list.append(np.mean(heg_pcc))
        hvg_pcc_list.append(np.mean(hvg_pcc))

        print(f"heg pcc: {np.mean(heg_pcc):.4f}")
        print(f"hvg pcc: {np.mean(hvg_pcc):.4f}")

    print("\nAverage Metrics:")
    print(f"avg heg pcc: {np.mean(heg_pcc_list):.4f}")
    print(f"avg hvg pcc: {np.mean(hvg_pcc_list):.4f}")
    print(f"avg MSE: {np.mean(mse_list):.4f}")
    print(f"avg MAE: {np.mean(mae_list):.4f}")
