import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import jensenshannon
from math import sqrt
import argparse
import os

parser = argparse.ArgumentParser(description='metrics')
parser.add_argument("--document", type=str, default='dataset45_ML')
args = parser.parse_args()

Data = args.document #
outdir = 'result/' + Data + '/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

def ssim(im1, im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2, "Both matrices must be 2-dimensional."
    assert im1.shape == im2.shape, "The shapes of the two matrices must match."

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

    ssim = l12 * c12 * s12

    return ssim

def calculate_ssim_for_genes(true_labels_df, predicted_labels_df):
    ssim_results = {}
    for gene in true_labels_df.columns:
        true_label = true_labels_df[gene].values.reshape(-1, 1)
        predicted_label = predicted_labels_df[gene].values.reshape(-1, 1)
        ssim_score = ssim(true_label, predicted_label)
        ssim_results[gene] = ssim_score
    return ssim_results

def calculate_pcc_per_gene(true_labels_df, predicted_labels_df):
    pcc_results = {gene: pearsonr(true_labels_df[gene], predicted_labels_df[gene])[0] for gene in true_labels_df.columns}
    return pcc_results

def calculate_rmse_per_gene(true_labels_df, predicted_labels_df):
    rmse_results = {gene: sqrt(mean_squared_error(true_labels_df[gene], predicted_labels_df[gene])) + 1 for gene in true_labels_df.columns}
    return rmse_results

def calculate_js_per_gene(true_labels_df, predicted_labels_df):
    js_results = {gene: jensenshannon(true_labels_df[gene], predicted_labels_df[gene]) for gene in true_labels_df.columns}
    return js_results

def calculate_as(detailed_results_df):
    detailed_results_df['Rank_PCC'] = detailed_results_df['PCC'].rank(ascending=False)
    detailed_results_df['Rank_SSIM'] = detailed_results_df['SSIM'].rank(ascending=False)
    detailed_results_df['Rank_RMSE'] = detailed_results_df['RMSE'].rank(ascending=True)
    detailed_results_df['Rank_JS'] = detailed_results_df['JS'].rank(ascending=True)

    detailed_results_df['AS'] = (detailed_results_df['Rank_PCC'] +
                                 detailed_results_df['Rank_SSIM'] +
                                 detailed_results_df['Rank_RMSE'] +
                                 detailed_results_df['Rank_JS']) / 4

    detailed_results_df['AS'] = detailed_results_df['AS'] / (detailed_results_df.shape[0]-1)
    return detailed_results_df

def main(true_labels_csv, predicted_values_csv):
    true_labels_df = pd.read_csv(true_labels_csv, index_col=0)
    predicted_labels_df = pd.read_csv(predicted_values_csv, index_col=0)

    true_labels_df = normalize_data(true_labels_df)
    predicted_labels_df = normalize_data(predicted_labels_df)

    ssim_per_gene = calculate_ssim_for_genes(true_labels_df, predicted_labels_df)
    pcc_per_gene = calculate_pcc_per_gene(true_labels_df, predicted_labels_df)
    rmse_per_gene = calculate_rmse_per_gene(true_labels_df, predicted_labels_df)
    js_per_gene = calculate_js_per_gene(true_labels_df, predicted_labels_df)

    detailed_results_df = pd.DataFrame({
        "PCC": pcc_per_gene,
        "SSIM": ssim_per_gene,
        "RMSE": rmse_per_gene,
        "JS": js_per_gene
    }).reset_index()
    detailed_results_df.rename(columns={'index': 'Gene'}, inplace=True)
    detailed_results_df.set_index("Gene", inplace=True)

    detailed_results_df = calculate_as(detailed_results_df)

    pcc_mean, pcc_std = np.mean(list(pcc_per_gene.values())), np.std(list(pcc_per_gene.values()))
    ssim_mean, ssim_std = np.mean(list(ssim_per_gene.values())), np.std(list(ssim_per_gene.values()))
    rmse_mean, rmse_std = np.mean(list(rmse_per_gene.values())), np.std(list(rmse_per_gene.values()))
    js_mean, js_std = np.mean(list(js_per_gene.values())), np.std(list(js_per_gene.values()))
    as_mean, as_std = np.mean(detailed_results_df['AS']), np.std(detailed_results_df['AS'])

    pcc_str = f"{pcc_mean:.3f}±{pcc_std:.3f}"
    ssim_str = f"{ssim_mean:.3f}±{ssim_std:.3f}"
    rmse_str = f"{rmse_mean:.3f}±{rmse_std:.3f}"
    js_str = f"{js_mean:.3f}±{js_std:.3f}"
    as_str = f"{as_mean:.3f}±{as_std:.3f}"

    mean_values_df = pd.DataFrame({
        "PCC": [pcc_str],
        "SSIM": [ssim_str],
        "RMSE": [rmse_str],
        "JS": [js_str],
        "AS": [as_str]
    }, index=["Mean Value"])

    return detailed_results_df, mean_values_df

true_labels_csv = outdir + 'original.csv'
predicted_values_csv = outdir + 'SpaDiT_prediction.csv'

detail_df, mean_df = main(true_labels_csv, predicted_values_csv)

detail_df.to_csv(outdir + '/' + Data + '_gene_result.csv', header=1, index=1)
mean_df.to_csv(outdir + '/' + Data + '_final_result.csv', header=1, index=1)
