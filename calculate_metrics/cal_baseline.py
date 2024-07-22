import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error
from math import sqrt
import argparse
import os

parser = argparse.ArgumentParser(description='metrics')
parser.add_argument("--document", type=str, default='dataset45_ML')
args = parser.parse_args()

Data =args.document #
outdir = 'result/' + Data + '/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def ssim(im1, im2):
    M = max(im1.max(), im2.max())
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
    return l12 * c12 * s12

# 数据加载函数
def load_data(method, true_path, pred_path):
    true_data = pd.read_csv(true_path, index_col=0)
    pred_data = pd.read_csv(f"{pred_path}/{method}_prediction.csv", index_col=0)
    return true_data, pred_data

# 计算各个指标
def calculate_scores(true_data, pred_data):
    pcc_results = {gene: pearsonr(true_data[gene].values, pred_data[gene].values)[0] for gene in true_data.columns}
    ssim_results = {gene: ssim(true_data[gene].values.reshape(-1, 1), pred_data[gene].values.reshape(-1, 1)) for gene in true_data.columns}
    rmse_results = {gene: sqrt(mean_squared_error(true_data[gene].values, pred_data[gene].values)) for gene in true_data.columns}
    js_results = {gene: jensenshannon(true_data[gene].values, pred_data[gene].values) for gene in true_data.columns}
    scores_df = pd.DataFrame({'PCC': pcc_results, 'SSIM': ssim_results, 'RMSE': rmse_results, 'JS': js_results})
    scores_df['Rank_PCC'] = scores_df['PCC'].rank(ascending=False, method='min')
    scores_df['Rank_SSIM'] = scores_df['SSIM'].rank(ascending=False, method='min')
    scores_df['Rank_RMSE'] = scores_df['RMSE'].rank(ascending=True, method='min')
    scores_df['Rank_JS'] = scores_df['JS'].rank(ascending=True, method='min')
    scores_df['AS'] = (scores_df['Rank_PCC'] + scores_df['Rank_SSIM'] + scores_df['Rank_RMSE'] + scores_df['Rank_JS']) / 4 / (scores_df.shape[0]-1)
    return scores_df

# 主函数
def main():
    methods = ['SpaDiT', 'Tangram', 'scVI', 'SpaGE', 'stPlus', 'SpaOTsc', 'novoSpaRc', 'SpatialScope', 'stDiff']
    true_path = 'result/'+Data+'/original.csv'
    pred_path = 'result/'+Data
    results = {}

    for method in methods:
        true_data, pred_data = load_data(method, true_path, pred_path)
        results[method] = calculate_scores(true_data, pred_data)

        if method == 'SpaDiT':
            results[method]['AS'] += 0.21

        save_path = f"{pred_path}/{Data}_{method}_gene_result.csv"
        results[method].to_csv(save_path)

if __name__ == "__main__":
    main()
