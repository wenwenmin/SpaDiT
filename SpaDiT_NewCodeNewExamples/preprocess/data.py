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


class ConditionalDiffusionDataset(Dataset):
    def __init__(self, sc_path, st_path, gene_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
        self.random_state = random_state
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        common_genes_df = pd.read_csv(gene_path + 'common_genes.csv')
        self.common_genes = common_genes_df['genes'].tolist()
        print(f"Number of shared genes: {len(self.common_genes)}")

        sc_adata = sc.read_h5ad(sc_path)
        st_adata = sc.read_h5ad(st_path)

        print(f"Original sc shape: {sc_adata.shape}")
        print(f"Original st shape: {st_adata.shape}")

        common_genes_in_sc = [g for g in self.common_genes if g in sc_adata.var_names]
        common_genes_in_st = [g for g in self.common_genes if g in st_adata.var_names]

        final_common_genes = list(set(common_genes_in_sc) & set(common_genes_in_st))
        print(f"Shared genes actually present in the data: {len(final_common_genes)}")

        sc_df = sc_adata[:, final_common_genes].to_df()
        st_df = st_adata[:, final_common_genes].to_df()

        print(f"sc shape (cells x genes): {sc_df.shape}")
        print(f"st shape (spots x genes): {st_df.shape}")

        self.gene_names = final_common_genes

        st_df_T = st_df.T
        sc_df_T = sc_df.T

        self.st_data = torch.tensor(st_df.values, dtype=torch.float32)

        self.st_sample = torch.tensor(st_df_T.values, dtype=torch.float32)
        self.sc_sample = torch.tensor(sc_df_T.values, dtype=torch.float32)
        self.sc_data = torch.tensor(sc_df.values, dtype=torch.float32)

        print(f"Shape after st transpose: {self.st_sample.shape}  (genes, spots)")
        print(f"Shape after sc transpose: {self.sc_sample.shape}  (genes, cells)")
        print(f"sc_data shape: {self.sc_data.shape}  (cells, genes)")

        np.random.seed(random_state)
        all_gene_indices = list(range(len(self.gene_names)))
        np.random.shuffle(all_gene_indices)

        n_genes = len(all_gene_indices)
        n_train = int(n_genes * train_ratio)
        n_val = int(n_genes * val_ratio)

        train_gene_indices = all_gene_indices[:n_train]
        val_gene_indices = all_gene_indices[n_train:n_train + n_val]
        test_gene_indices = all_gene_indices[n_train + n_val:]

        self.train_gene_indices = train_gene_indices
        self.val_gene_indices = val_gene_indices
        self.test_gene_indices = test_gene_indices

        print(f"\nGene split:")
        print(f"  Train genes: {len(train_gene_indices)} ({train_ratio*100:.0f}%)")
        print(f"  Validation genes: {len(val_gene_indices)} ({val_ratio*100:.0f}%)")
        print(f"  Test genes: {len(test_gene_indices)} ({test_ratio*100:.0f}%)")
        print(f"  Train genes: {[self.gene_names[i] for i in train_gene_indices[:5]]}...")
        print(f"  Test genes: {[self.gene_names[i] for i in test_gene_indices[:5]]}...")

    def __len__(self):
        return len(self.gene_names)

    def __getitem__(self, idx):
        st_gene = self.st_sample[idx]
        sc_gene = self.sc_sample[idx]
        return st_gene, sc_gene, self.sc_data

    def get_sc_data(self):
        return self.sc_data

    def get_gene_names(self):
        return self.gene_names

    def get_train_gene_indices(self):
        return self.train_gene_indices

    def get_val_gene_indices(self):
        return self.val_gene_indices

    def get_test_gene_indices(self):
        return self.test_gene_indices


class MaskedDataset(Dataset):
    def __init__(self, base_dataset, gene_mask):
        self.base_dataset = base_dataset
        self.gene_mask = gene_mask
        self.n_genes = gene_mask.sum().item()

    def __len__(self):
        return self.n_genes

    def __getitem__(self, idx):
        masked_indices = torch.where(self.gene_mask)[0]
        gene_idx = masked_indices[idx]

        st_gene, sc_gene, sc_data = self.base_dataset[gene_idx.item()]
        return st_gene, sc_gene, sc_data


def collate_with_condition(batch):
    st_samples = []
    sc_samples = []
    sc_conditions = []

    for st, sc, sc_cond in batch:
        st_samples.append(st)
        sc_samples.append(sc)
        sc_conditions.append(sc_cond)

    x = torch.stack(st_samples, 0)
    x_hat = torch.stack(sc_samples, 0)
    y = sc_conditions[0]

    return x, x_hat, y


def reindex(adata, genes, chunk_size=CHUNK_SIZE):
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
    sc.set_figure_params(dpi=80, figsize=(3, 3))
    hvg_adata = hvg_adata.copy()
    if save_filename:
        sc.settings.figdir = save_filename
        save = '.pdf'
    else:
        save = None

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
        generator=generator)


def scale(adata):
    scaler = MaxAbsScaler()
    normalized_data = scaler.fit_transform(adata.X.T).T

    adata.X = normalized_data
    return adata


def data_augment(adata: AnnData, fixed: bool, noise_std):
    noise_stddev = noise_std
    augmented_adata = adata.copy()
    gene_expression = adata.X

    if fixed:
        augmented_adata.X = augmented_adata.X + np.full(gene_expression.shape, noise_stddev)
    else:
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_stddev, gene_expression.shape))

    merge_adata = adata.concatenate(augmented_adata, join='outer')

    return merge_adata


