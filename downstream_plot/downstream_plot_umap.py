import scanpy as sc
import pandas as pd
import os
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser(description='Plot UMAPs for multiple prediction methods')
parser.add_argument('--document', type=str, default='dataset40_MC')
args = parser.parse_args()
data_set = args.document

data_path = 'result/' + data_set
save_path = 'downstream_result/umap/' + data_set
if not os.path.exists(save_path):
    os.makedirs(save_path)

methods = ['SpaDiT', 'Tangram', 'scVI', 'SpaGE', 'stPlus', 'SpaOTsc', 'novoSpaRc', 'SpatialScope', 'stDiff']

def plot_hvg_umap(hvg_adata, ax, color, legend=True):
    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata)
    sc.pp.neighbors(hvg_adata, n_pcs=30, n_neighbors=30)
    sc.tl.umap(hvg_adata, min_dist=0.1)
    sc.pl.umap(hvg_adata, color=color, legend_fontsize=12, ncols=2, show=False, ax=ax)
def plot_all_methods(methods, data_path, save_path):
    for method in methods:
        predict_path = os.path.join(data_path, f'{method}_prediction.csv')
        origin_path = os.path.join(data_path, 'original.csv')
        if not os.path.isfile(predict_path) or not os.path.isfile(origin_path):
            continue

        predict_data = pd.read_csv(predict_path, header=0, index_col=0)
        origin_data = pd.read_csv(origin_path, header=0, index_col=0)

        obs = pd.DataFrame(index=predict_data.index)
        var = pd.DataFrame(index=predict_data.columns)
        pred_share = sc.AnnData(X=predict_data.values, obs=obs, var=var)
        origin_share = sc.AnnData(X=origin_data.values, obs=obs, var=var)

        pred_share.obs[method] = 'ST_pred'
        pred_share.obs[method] = pred_share.obs[method].astype('category')
        origin_share.obs[method] = 'ST_Original'
        origin_share.obs[method] = origin_share.obs[method].astype('category')

        merged_data = sc.concat([pred_share, origin_share], axis=0)

        fig, ax = plt.subplots(figsize=(6, 5))
        plot_hvg_umap(merged_data, ax, color=[method])
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, data_set+f'_{method}_umap.png'), format='png')
        plt.close(fig)

# 运行绘图函数
plot_all_methods(methods, data_path, save_path)
