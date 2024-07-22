import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
import scipy
import os
from preprocess.data import reindex
from utils import calculate_zero_percentage, ensure_dir_exists
import argparse
parser = argparse.ArgumentParser(description='Process some datasets')
parser.add_argument('--sc_data', type=str, default='scRNA_count.txt')
parser.add_argument('--st_data', type=str, default='Spatial_count.txt')
parser.add_argument('--document', type=str, default='dataset40_MC')
args = parser.parse_args()
print(os.getcwd())

sc_path = '../datasets/' + args.document + '/' + args.sc_data
st_path = '../datasets/' + args.document + '/' + args.st_data
adata_seq = sc.read(sc_path, sep='\t', first_column_names=True).T
adata_spatial = sc.read(st_path, sep='\t')
print('sc dataset shape:', adata_seq.shape)
print('st dataset shape:', adata_spatial.shape)
adata_seq_copy = adata_seq.copy()
adata_spatial_copy = adata_spatial.copy()

sc.pp.normalize_total(adata_seq_copy, target_sum=1e4)
sc.pp.log1p(adata_seq_copy)
sc.pp.filter_genes(adata_seq_copy, min_cells=int(adata_seq_copy.shape[0] * 0.1))
sc.pp.filter_cells(adata_seq_copy, min_genes=200)
# sc.pp.highly_variable_genes(adata_seq_copy, n_top_genes=2000)
# adata_seq_copy = adata_seq_copy[:, adata_seq_copy.var.highly_variable]
sc.pp.highly_variable_genes(adata_seq_copy, n_top_genes=int(0.25 * adata_seq_copy.shape[1]))
adata_seq_copy = adata_seq_copy[:, adata_seq_copy.var.highly_variable]

sc.pp.normalize_total(adata_spatial_copy, target_sum=1e4)
sc.pp.log1p(adata_spatial_copy)
sc.pp.filter_genes(adata_spatial_copy, min_cells=1)
sc.pp.filter_genes(adata_spatial_copy, min_cells=int(adata_spatial_copy.shape[0] * 0.1))
sc.pp.filter_cells(adata_spatial_copy, min_genes=1)
sc.pp.highly_variable_genes(adata_spatial_copy, n_top_genes=int(0.25 * adata_seq_copy.shape[1]))
adata_spatial_copy = adata_spatial_copy[:, adata_spatial_copy.var.highly_variable]

sc_common_genes = list(set(adata_seq_copy.var_names).intersection(set(adata_spatial_copy.var_names)))
unique_to_sc = list(set(adata_seq_copy.var_names) - set(sc_common_genes))
sc_existing_genes = [gene for gene in sc_common_genes + unique_to_sc if gene in adata_seq_copy.var_names]
adata_seq_common = adata_seq_copy[:, sc_existing_genes]

st_common_genes = sc_common_genes
unique_to_st = list(set(adata_spatial_copy.var_names) - set(st_common_genes))
st_existing_genes = [gene for gene in st_common_genes + unique_to_st if gene in adata_spatial_copy.var_names]
adata_spatial_common = adata_spatial_copy[:,st_existing_genes]

print('sc dataset shape (filtered, common genes):', adata_seq_common.shape)
print('st dataset shape (filtered, common genes):', adata_spatial_common.shape)
print(f"sc dataset zero percent: {calculate_zero_percentage(adata_seq_common):.2f}%")
print(f"st dataset zero percent: {calculate_zero_percentage(adata_spatial_common):.2f}%")

sc_data_save = '../datasets/' + args.document + '/sc/'
st_data_save = '../datasets/' + args.document + '/st/'
gene_name_save = '../datasets/' + args.document + '/gene/'

ensure_dir_exists(sc_data_save)
ensure_dir_exists(st_data_save)
ensure_dir_exists(gene_name_save)

adata_seq_common.write(os.path.join(sc_data_save, args.document + '_sc.h5ad'))
adata_spatial_common.write(os.path.join(st_data_save, args.document + '_st.h5ad'))
pd.DataFrame(sc_common_genes, columns=['genes']).to_csv(os.path.join(gene_name_save, 'common_genes.csv'), index=False)
pd.DataFrame(unique_to_sc, columns=['genes']).to_csv(os.path.join(gene_name_save, 'unique_to_sc.csv'), index=False)
pd.DataFrame(unique_to_st, columns=['genes']).to_csv(os.path.join(gene_name_save, 'unique_to_st.csv'), index=False)

adata_seq_common.write(os.path.join(sc_data_save, args.document + '_sc.h5ad'))
adata_spatial_common.write(os.path.join(st_data_save, args.document + '_st.h5ad'))