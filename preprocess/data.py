import scipy
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from scipy.sparse import issparse, csr
from anndata import AnnData
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
CHUNK_SIZE = 20000

# class ConditionalDiffusionDataset(Dataset):
#     def __init__(self, sc_path, st_path, transform=None):
#         # 读取数据
#         self.sc_data = sc.read_h5ad(sc_path)
#         self.st_data = sc.read_h5ad(st_path)
#         self.transform = transform
#
#         self.max_len = max(len(self.sc_data), len(self.st_data))
#         sc_df = self.sc_data.to_df()
#         st_df = self.st_data.to_df()
#
#         if len(sc_df) < self.max_len:
#             sc_df = sc_df.sample(n=self.max_len, replace=True)
#         if len(st_df) < self.max_len:
#             st_df = st_df.sample(n=self.max_len, replace=True)
#
#         sc_df = sc_df.iloc[:self.max_len]
#         st_df = st_df.iloc[:self.max_len]
#
#         self.sc_data = sc_df
#         self.st_data = st_df
#
#     def __len__(self):
#         return self.max_len
#
#     def __getitem__(self, idx):
#         sc_sample = self.sc_data.iloc[idx]
#         st_sample = self.st_data.iloc[idx]
#         if self.transform:
#             sc_sample, st_sample = self.transform(sc_sample, st_sample)
#         sc_sample = torch.tensor(sc_sample.values, dtype=torch.float32)
#         st_sample = torch.tensor(st_sample.values, dtype=torch.float32)
#
#         return st_sample, sc_sample

class ConditionalDiffusionDataset(Dataset):
    def __init__(self, sc_path, st_path):
        self.sc_data = sc.read_h5ad(sc_path)
        self.st_data = sc.read_h5ad(st_path)
        self.st_data = self.st_data.to_df().T
        self.sc_data = self.sc_data.to_df().T


        self.gene_names = self.st_data.index.tolist()

        self.st_sample = torch.tensor(self.st_data.values, dtype=torch.float32)
        self.sc_sample = torch.tensor(self.sc_data.values, dtype=torch.float32)
        self.sc_data = torch.tensor(self.sc_data.values, dtype=torch.float32)

    def __len__(self):
        return len(self.st_data)

    def __getitem__(self, idx):
        return self.st_sample[idx], self.sc_sample[idx], self.sc_data

    def get_gene_names(self):
        return self.gene_names



def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list

    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing
    chunk_size
        chunk large data into small chunks

    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        for i in range(new_X.shape[0] // chunk_size + 1):
            new_X[i * chunk_size:(i + 1) * chunk_size, idx] = adata[i * chunk_size:(i + 1) * chunk_size, genes[idx]].X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={'var_names': genes})
    return adata


def plot_hvg_umap(hvg_adata, color=['celltype'], save_filename=None):
    sc.set_figure_params(dpi=80, figsize=(3, 3))  # type: ignore
    hvg_adata = hvg_adata.copy()
    if save_filename:
        sc.settings.figdir = save_filename
        save = '.pdf'
    else:
        save = None
    # ideal gas equation

    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata)
    sc.pp.neighbors(hvg_adata, n_pcs=30, n_neighbors=30)
    sc.tl.umap(hvg_adata, min_dist=0.1)
    sc.pl.umap(hvg_adata, color=color, legend_fontsize=10, ncols=2, show=None, save=save, wspace=1)
    return hvg_adata


def get_data_loader(data_ary: np.ndarray,
                    cell_type: np.ndarray,
                    batch_size: int = 512,
                    is_shuffle: bool = True,
                    ):
    data_tensor = torch.from_numpy(data_ary.astype(np.float32))
    cell_type_tensor = torch.from_numpy(cell_type.astype(np.float32))
    dataset = TensorDataset(data_tensor, cell_type_tensor)
    generator = torch.Generator(device='cuda')
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False,
        generator=generator)  # , generator=torch.Generator(device = 'cuda')


def scale(adata):
    scaler = MaxAbsScaler()
    # 对adata.X按行进行归一化
    normalized_data = scaler.fit_transform(adata.X.T).T

    # 更新归一化后的数据到adata.X
    adata.X = normalized_data
    return adata


def data_augment(adata: AnnData, fixed: bool, noise_std):
    # 定义增强参数，例如噪声的标准差
    noise_stddev = noise_std
    augmented_adata = adata.copy()
    gene_expression = adata.X

    if fixed:
        augmented_adata.X = augmented_adata.X + np.full(gene_expression.shape, noise_stddev)
    else:
        # 对每个基因的表达值引入随机噪声
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_stddev, gene_expression.shape))

    merge_adata = adata.concatenate(augmented_adata, join='outer')

    return merge_adata




