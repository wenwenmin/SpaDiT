import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='RS_Scatter')
parser.add_argument('--document', type=str, default='DS_05')
args = parser.parse_args()
data_file = args.document

methods = ['SpaDiT', 'Tangram', 'scVI', 'SpaGE', 'stPlus', 'SpaOTsc', 'novoSpaRc', 'SpatialScope', 'stDiff']

for method in methods:

    file_path = f"downstream_result/RS_Scatter/data/"+data_file+f"/{method}_data.csv"
    df = pd.read_csv(file_path)
    original = df['original']
    downsampled = df['downsampled']


    fig, ax = plt.subplots(figsize=(6, 4))

    rs = np.mean((original > 0.5) & (downsampled > 0.5))


    ax.scatter(original, downsampled, color='gray', s=10)
    ax.scatter(original[(original > 0.5) & (downsampled > 0.5)],
               downsampled[(original > 0.5) & (downsampled > 0.5)], color='red', s=10)
    ax.set_title(f'{method}\nRS={rs:.2f}')
    ax.set_xlabel('PCC (Groundtruth)')
    ax.set_ylabel('PCC (Downsample)')

    plt.savefig(f"downstream_result/RS_Scatter/fig/"+data_file+f"/{method}.pdf")
    plt.close()

print("Plots generated and saved successfully for all methods.")
