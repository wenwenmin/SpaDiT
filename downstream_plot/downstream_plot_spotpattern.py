import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
import argparse


warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='spot-spot Distance Heatmap for Various Methods')
parser.add_argument('--document', type=str, default='dataset40_MC')
args = parser.parse_args()
data_set = args.document


data_path = 'result/' + data_set
save_path = 'downstream_result/heatmap_spot/' + data_set
if not os.path.exists(save_path):
    os.makedirs(save_path)


cmap_blue_gold = LinearSegmentedColormap.from_list(
    'custom_blue_gold',
    ['#023858', '#045a8d', '#0570b0', '#3690c0', '#74a9cf', '#a6bddb', '#d0d1e6', '#fee391', '#fec44f', '#fe9929',
     '#ec7014', '#cc4c02', '#993404', '#662506']
)

methods = ['SpaDiT', 'Tangram', 'scVI', 'SpaGE', 'stPlus', 'SpaOTsc', 'novoSpaRc', 'SpatialScope', 'stDiff']

def plot_gene_gene_heatmap(file_path, title, save_path):
    try:
        data = pd.read_csv(file_path, header=0, index_col=0)
        data = data.T
        if data.empty:
            print(f"No data to plot for {title}.")
            return

        gene_distances = pdist(data.T, metric='euclidean')
        distance_matrix = squareform(gene_distances)

        linkage_matrix = linkage(gene_distances, method='complete')
        dendro = dendrogram(linkage_matrix, no_plot=True)
        order = dendro['leaves']

        sorted_matrix = distance_matrix[order, :][:, order]

        plt.figure(figsize=(10, 8))
        sns.heatmap(sorted_matrix, cmap=cmap_blue_gold, xticklabels=False, yticklabels=False)
        plt.title(title)
        plt.savefig(os.path.join(save_path, title + '.png'))
        plt.close()
    except Exception as e:
        print(f"An error occurred while processing {title}: {str(e)}")


def main():
    try:
        original_path = os.path.join(data_path, 'original.csv')
        plot_gene_gene_heatmap(original_path, f'{data_set}_original', save_path)
        print("Original plot completed.")

        for method in methods:
            file_path = os.path.join(data_path, f'{method}_prediction.csv')
            if not os.path.isfile(file_path):
                print(f"Data file not found for method {method}: {file_path}")
                continue
            plot_gene_gene_heatmap(file_path, f'{data_set}_{method}', save_path)
            print(f"Plot for {method} completed.")
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()

