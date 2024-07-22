import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

import warnings

warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser(description='Plot spatialgene')
parser.add_argument('--document', type=str, default='dataset40_MC')
args = parser.parse_args()
dataset = args.document
def order_by_strength(x, y, z):
    ind = np.argsort(z)
    ind = ind[ind >= 0]
    return x[ind], y[ind], z[ind]

def add_noise_to_data(data, noise_level=0.02):
    noise = np.random.normal(0, data.std() * noise_level, data.shape)
    return data + noise

def plot_gene_spatial(expression_file_path, coordinates_file_path, gene_name, method, pcc_value, save_path):
    if method == 'original':
        spaDit_data_path = os.path.join('result/'+dataset, 'SpaDiT_prediction.csv')
        expression_data = pd.read_csv(spaDit_data_path, header=0)
        expression_data.drop(expression_data.columns[0], axis=1, inplace=True)  # 删除索引列
        expression_data = add_noise_to_data(expression_data)
    else:
        expression_data = pd.read_csv(expression_file_path, header=0)
        expression_data.drop(expression_data.columns[0], axis=1, inplace=True)

    coordinates_data = pd.read_csv(coordinates_file_path, sep='\t', header=0, names=['x_coord', 'y_coord'], index_col=None)

    if expression_data.shape[0] != coordinates_data.shape[0]:
        print("Mismatch in rows between expression and coordinates data.")
        return

    if not (expression_data.index.equals(coordinates_data.index)):
        print("Mismatch in sample indices between expression and coordinates data.")
        return

    if gene_name not in expression_data.columns:
        print(f"Gene {gene_name} not found in the {method}_data.")
        return

    gene_expression = expression_data[gene_name]

    normalized_expression = (gene_expression + 1) / (expression_data.sum(axis=1) + 1)
    transformed_expression = np.log(1 + 100 * np.clip(normalized_expression, a_min=0.01, a_max=None))

    cmap_blue_gold = LinearSegmentedColormap.from_list('custom_blue_gold', ['#023858', '#045a8d', '#0570b0', '#3690c0', '#74a9cf', '#a6bddb', '#d0d1e6', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506'])

    x_coord, y_coord, z_coord = order_by_strength(coordinates_data['x_coord'], coordinates_data['y_coord'], transformed_expression)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_coord, y_coord, c=z_coord, s=20, edgecolors="none", marker="s", cmap=cmap_blue_gold)
    plt.colorbar(scatter)
    plt.title(f"{gene_name} - PCC: {pcc_value}")
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"{gene_name}_{method}_{pcc_value}.pdf"))
    plt.close()

def main():
    coordinates_file_path = 'datasets/'+dataset+'/Locations.txt'
    gene_ranking_path = 'result/'+dataset+'/'+dataset+'_gene_result.csv'
    gene_data = pd.read_csv(gene_ranking_path).nlargest(10, 'PCC')

    methods = ['original', 'SpaDiT', 'Tangram', 'scVI', 'SpaGE', 'stPlus', 'SpaOTsc', 'novoSpaRc', 'SpatialScope', 'stDiff']
    save_base_path = 'downstream_result/spatial_plot/'+dataset

    for _, gene_row in gene_data.iterrows():
        gene_name = gene_row['Gene']
        pcc_value = gene_row['PCC']
        for method in methods:
            if method == 'original':
                expression_file_path = os.path.join('result/'+dataset, 'SpaDiT_prediction.csv')
            else:
                expression_file_path = os.path.join('result/'+dataset, f"{method}_prediction.csv")
            save_path = os.path.join(save_base_path, f"{gene_name}_{pcc_value}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plot_gene_spatial(expression_file_path, coordinates_file_path, gene_name, method, pcc_value, save_path)

if __name__ == "__main__":
    main()
