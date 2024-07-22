import pandas as pd
import numpy as np
import scanpy as sc
import scipy.stats as st
import os
import seaborn as sns
import scipy
from scipy.sparse import csr_matrix
import torch
import random
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, \
    normalized_mutual_info_score
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split


def pca_with_torch(X, k=100):
    X_mean = torch.mean(X, dim=0)
    X = X - X_mean

    cov_matrix = torch.matmul(X.t(), X) / (X.size(0) - 1)

    L_complex, V_complex = torch.linalg.eig(cov_matrix)

    eigenvalues = L_complex.real
    eigenvectors = V_complex.real
    _, indices = torch.sort(eigenvalues, descending=True)
    components = eigenvectors[:, indices[:k]]
    X_pca = torch.matmul(X, components)

    return X_pca

def mask_tensor_with_masks(X, mask_zero_ratio, mask_nonzero_ratio, device='cuda:0'):
    X = X.to(device)
    nonzero_indices = torch.nonzero(X, as_tuple=True)
    num_nonzero_to_mask = int(round(mask_nonzero_ratio * len(nonzero_indices[0])))
    nonzero_mask_indices = torch.randperm(len(nonzero_indices[0]))[:num_nonzero_to_mask]

    zero_indices = torch.nonzero(X == 0, as_tuple=True)
    num_zero_to_mask = int(round(mask_zero_ratio * len(zero_indices[0])))
    zero_mask_indices = torch.randperm(len(zero_indices[0]))[:num_zero_to_mask]

    masked_X = X.clone()
    masked_X[nonzero_indices[0][nonzero_mask_indices], nonzero_indices[1][nonzero_mask_indices]] = 0
    masked_X[zero_indices[0][zero_mask_indices], zero_indices[1][zero_mask_indices]] = 0

    M_nonzero_mask = torch.zeros_like(X, dtype=torch.float32)
    M_nonzero_mask[nonzero_indices[0][nonzero_mask_indices], nonzero_indices[1][nonzero_mask_indices]] = 1.0

    M_zero_mask = torch.zeros_like(X, dtype=torch.float32)
    M_zero_mask[zero_indices[0][zero_mask_indices], zero_indices[1][zero_mask_indices]] = 1.0

    return masked_X, M_nonzero_mask, M_zero_mask

def clustering_metrics(adata, target, pred, mode="AMI"):
    """
    Evaluate clustering performance.
   
    Parameters
    ----------
    adata
        AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    target
        Key in `adata.obs` where ground-truth spatial domain labels are stored.
    pred
        Key in `adata.obs` where clustering assignments are stored.
       
    Returns
    -------
    ami
        Adjusted mutual information score.
    ari
        Adjusted Rand index score.
    homo
        Homogeneity score.
    nmi
        Normalized mutual information score.

    """
    if(mode=="AMI"):
        ami = adjusted_mutual_info_score(adata.obs[target], adata.obs[pred])
        print("AMI ",ami)
        return ami
    elif(mode=="ARI"):
        ari = adjusted_rand_score(adata.obs[target], adata.obs[pred])
        print("ARI ",ari)
        return ari
    elif(mode=="Homo"):
        homo = homogeneity_score(adata.obs[target], adata.obs[pred])
        print("Homo ",homo)
        return homo
    elif(mode=="NMI"):
        nmi = normalized_mutual_info_score(adata.obs[target], adata.obs[pred])
        print("NMI ", nmi)
        return nmi

def calculate_pcc(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape."
    mean1 = tensor1.mean(dim=0)
    mean2 = tensor2.mean(dim=0)
    covariance = ((tensor1 - mean1) * (tensor2 - mean2)).mean(dim=0)
    variance1 = ((tensor1 - mean1) ** 2).mean(dim=0)
    variance2 = ((tensor2 - mean2) ** 2).mean(dim=0)
    pcc = covariance / torch.sqrt(variance1 * variance2)

    return pcc.mean()

def calculate_pcc_with_mask(tensor1, tensor2, mask):
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape."
    assert tensor1.shape == mask.shape, "Mask tensor must have the same shape as input tensors."

    mask = mask.bool()
    # 仅考虑mask为True的部分
    masked_tensor1 = tensor1[mask]
    masked_tensor2 = tensor2[mask]

    mean1 = masked_tensor1.mean()
    mean2 = masked_tensor2.mean()

    covariance = ((masked_tensor1 - mean1) * (masked_tensor2 - mean2)).mean()
    variance1 = ((masked_tensor1 - mean1) ** 2).mean()
    variance2 = ((masked_tensor2 - mean2) ** 2).mean()

    pcc = covariance / torch.sqrt(variance1 * variance2)

    return pcc

def calculate_rmse(tensor1, tensor2):
    squared_diff = (tensor1 - tensor2) ** 2
    mean_squared_diff = squared_diff.mean()
    rmse = torch.sqrt(mean_squared_diff)
    return rmse

def calculate_rmse_with_mask(tensor1, tensor2, mask):
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape."
    assert tensor1.shape == mask.shape, "Mask tensor must have the same shape as input tensors."

    mask = mask.bool()
    # 仅考虑mask为True的部分
    masked_tensor1 = tensor1[mask]
    masked_tensor2 = tensor2[mask]

    squared_diff = (masked_tensor1 - masked_tensor2) ** 2
    mean_squared_diff = squared_diff.mean()
    rmse = torch.sqrt(mean_squared_diff)

    return rmse


def calculate_pcc_per_gene(true_labels, predicted_labels):
    pcc_sum = 0.0
    for gene in range(true_labels.shape[1]):
        y_true = true_labels[:, gene]
        y_pred = predicted_labels[:, gene]

        mean_true = y_true.mean()
        mean_pred = y_pred.mean()

        covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
        variance_true = ((y_true - mean_true) ** 2).mean()
        variance_pred = ((y_pred - mean_pred) ** 2).mean()

        pcc = covariance / torch.sqrt(variance_true * variance_pred)
        pcc_sum += pcc

    return (pcc_sum / true_labels.shape[1]).item()


def calculate_rmse_per_gene(true_labels, predicted_labels):
    mse_sum = 0.0
    for gene in range(true_labels.shape[1]):
        y_true = true_labels[:, gene]
        y_pred = predicted_labels[:, gene]

        mse = torch.mean((y_true - y_pred) ** 2)
        mse_sum += mse

    return torch.sqrt(mse_sum / true_labels.shape[1]).item()

def split_dataset_with_gene_names(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=None):

    total_size = len(dataset)
    indices = list(range(total_size))

    train_indices, temp_indices = train_test_split(indices, train_size=train_ratio, random_state=random_state)

    remaining_ratio = 1 - train_ratio  # 0.3
    val_ratio_adjusted = val_ratio / remaining_ratio
    val_indices, test_indices = train_test_split(temp_indices, train_size=val_ratio_adjusted, random_state=random_state)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_gene_names = [dataset.get_gene_names()[i] for i in train_indices]
    val_gene_names = [dataset.get_gene_names()[i] for i in val_indices]
    test_gene_names = [dataset.get_gene_names()[i] for i in test_indices]

    return (train_dataset, train_gene_names), (val_dataset, val_gene_names), (test_dataset, test_gene_names)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_zero_percentage(adata):
    data_matrix = adata.X
    total_elements = data_matrix.shape[0] * data_matrix.shape[1]

    if scipy.sparse.issparse(data_matrix):
        nonzero_elements = data_matrix.count_nonzero()
        zero_elements = total_elements - nonzero_elements
    else:
        zero_elements = np.count_nonzero(data_matrix == 0)

    zero_percentage = (zero_elements / total_elements) * 100
    return zero_percentage

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def add_constant_to_sparse_matrix(adata):
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()  # Convert to dense
    adata.X += 1e-10  # Add small constant
    adata.X = csr_matrix(adata.X)  # Convert back to sparse

    return adata


def cal_ssim(im1, im2, M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12

    return ssim


def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.max()
        result = pd.concat([result, content], axis=1)
    return result


def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = st.zscore(content)
        content = pd.DataFrame(content, columns=[label])
        result = pd.concat([result, content], axis=1)
    return result


def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.sum()
        result = pd.concat([result, content], axis=1)
    return result


def logNorm(df):
    df = np.log1p(df)
    df = st.zscore(df)
    return df


class CalculateMeteics:
    def __init__(self, raw_data, adata_data, genes_name, impute_count_file, prefix, metric):
        self.impute_count_file = impute_count_file
        self.raw_count = pd.DataFrame(raw_data, columns=genes_name)
        self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
        self.raw_count = self.raw_count.T
        self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        self.raw_count = self.raw_count.fillna(1e-20)

        self.adata_data = adata_data

        self.impute_count = pd.read_csv(impute_count_file, header=0, index_col=0)
        self.impute_count.columns = [x.upper() for x in self.impute_count.columns]
        self.impute_count = self.impute_count.T
        self.impute_count = self.impute_count.loc[~self.impute_count.index.duplicated(keep='first')].T
        self.impute_count = self.impute_count.fillna(1e-20)
        self.prefix = prefix
        self.metric = metric

    def SSIM(self, raw, impute, scale='scale_max'):
        print('---------Calculating SSIM---------')
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print('Please note you do not scale data by scale max')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    ssim = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    M = [raw_col.max(), impute_col.max()][raw_col.max() > impute_col.max()]
                    raw_col_2 = np.array(raw_col)
                    raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0], 1)
                    impute_col_2 = np.array(impute_col)
                    impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0], 1)
                    ssim = cal_ssim(raw_col_2, impute_col_2, M)

                ssim_df = pd.DataFrame(ssim, index=["SSIM"], columns=[label])
                result = pd.concat([result, ssim_df], axis=1)
        else:
            print("columns error")
            return pd.DataFrame()

        print(result)
        return result

    def PCC(self, raw, impute, scale = 'scale_z_score'):
        print('---------Calculating PCC---------')
        if scale == 'scale_z_score':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print('Please note you do not scale data by logNorm')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    pearsonr = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e2)
                    raw_col = raw_col.fillna(1e2)
                    pearsonr, _ = st.pearsonr(raw_col, impute_col)
                pcc_df = pd.DataFrame(pearsonr, index=["PCC"], columns=[label])
                result = pd.concat([result, pcc_df], axis=1)
        else:
            print("columns error")

        print(result)
        return result

    def JS(self, raw, impute, scale='scale_plus'):
        print('---------Calculating JS---------')
        if scale == 'scale_plus':
            raw = scale_plus(raw)
            impute = scale_plus(impute)
        else:
            print('Please note you do not scale data by plus')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    JS = 1
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    raw_col = raw_col.fillna(1e-20)
                    impute_col = impute_col.fillna(1e-20)
                    M = (raw_col + impute_col) / 2
                    JS = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)
                JS_df = pd.DataFrame(JS, index=["JS"], columns=[label])
                result = pd.concat([result, JS_df], axis=1)
        else:
            print("columns error")

        print(result)
        return result

    def RMSE(self, raw, impute, scale='zscore'):
        print('---------Calculating RMSE---------')
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print('Please note you do not scale data by zscore')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    RMSE = 1.5
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"], columns=[label])
                result = pd.concat([result, RMSE_df], axis=1)
        else:
            print("columns error")

        print(result)
        return result

    def cluster(self, adata_data, impu, scale=None):
        print('---------Calculating cluster---------')

        cpy_x = adata_data.copy()
        cpy_x.X = impu

        sc.tl.pca(adata_data)
        sc.pp.neighbors(adata_data, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(adata_data)
        tmp_adata1 = adata_data

        sc.tl.pca(cpy_x)
        sc.pp.neighbors(cpy_x, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(cpy_x)
        tmp_adata2 = cpy_x

        tmp_adata2.obs['class'] = tmp_adata1.obs['leiden']

        # tmp_adata2 = get_N_clusters(cpy_x, 23, 'leiden') # merfish-mop 23类别
        # tmp_adata2.obs['class'] = adata_spatial2.obs['subclass_label']

        ari = clustering_metrics(tmp_adata2, 'class', 'leiden', "ARI")
        ami = clustering_metrics(tmp_adata2, 'class', 'leiden', "AMI")
        homo = clustering_metrics(tmp_adata2, 'class', 'leiden', "Homo")
        nmi = clustering_metrics(tmp_adata2, 'class', 'leiden', "NMI")
        result = pd.DataFrame([[ari, ami, homo, nmi]], columns=["ARI", "AMI", "Homo", "NMI"])
        return result

    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count
        prefix = self.prefix
        adata_data = self.adata_data
        SSIM_gene = self.SSIM(raw, impute)
        # Spearman_gene = self.SPCC(raw, impute)
        PCC_gene = self.PCC(raw, impute)
        JS_gene = self.JS(raw, impute)
        RMSE_gene = self.RMSE(raw, impute)

        cluster_result = self.cluster(adata_data, impute)

        result_gene = pd.concat([PCC_gene, SSIM_gene, RMSE_gene, JS_gene], axis=0)
        result_gene.T.to_csv(prefix + "_gene_Metrics.txt", sep='\t', header=1, index=1)

        cluster_result.to_csv(prefix + "_cluster_Metrics.txt", sep='\t', header=1, index=1)

        return result_gene


def CalDataMetric(Data, PATH, sp_data, sp_genes, adata_data, out_dir):
    print('We are calculating the : ' + Data + '\n')
    metrics = ['PCC(gene)',  'SSIM(gene)', 'RMSE(gene)', 'JS(gene)']
    metric = ['PCC', 'SSIM', 'RMSE', 'JS']
    impute_count_dir = PATH + Data
    impute_count = os.listdir(impute_count_dir)
    impute_count = [x for x in impute_count if x[-3:] == 'csv' and x != 'final_result.csv']
    methods = []
    if len(impute_count) != 0:
        medians = pd.DataFrame()
        for impute_count_file in impute_count:
            print(impute_count_file)
            if 'result_Tangram.csv' in impute_count_file:
                os.system('mv ' + impute_count_dir + '/result_Tangram.csv ' + impute_count_dir + '/Tangram_impute.csv')
            prefix = impute_count_file.split('_')[0]
            methods.append(prefix)
            prefix = impute_count_dir + '/' + prefix
            impute_count_file = impute_count_dir + '/' + impute_count_file
            # if not os.path.isfile(prefix + '_Metrics.txt'):
            print(impute_count_file)
            CM = CalculateMeteics(sp_data, adata_data, sp_genes, impute_count_file=impute_count_file, prefix=prefix,
                                  metric=metric)
            CM.compute_all()

            # 计算中位数
            median = []
            for j in ['_gene']:
                # j = '_gene'
                #     median = []
                tmp = pd.read_csv(prefix + j + '_Metrics.txt', sep='\t', index_col=0)
                for m in metric:
                    median.append(np.mean(tmp[m]))
                    # median.append((np.max(tmp[m]) + np.min(tmp[m]))/2)
            median = pd.DataFrame([median], columns=metrics)
            # 聚类指标

            clu = pd.read_csv(prefix + '_cluster' + '_Metrics.txt', sep='\t', index_col=0)
            median = pd.concat([median, clu], axis=1)
            medians = pd.concat([medians, median], axis=0)


        metrics += ["ARI", "AMI", "Homo", "NMI"]
        medians.columns = metrics
        medians.index = methods
        medians.to_csv(out_dir + '/final_result.csv', header=1, index=1)