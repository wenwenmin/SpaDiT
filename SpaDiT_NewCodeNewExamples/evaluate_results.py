import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def calculate_overall_pcc(pred, gt):
    pcc, _ = pearsonr(pred.flatten(), gt.flatten())
    return pcc


def calculate_rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))


def evaluate_predictions(pred_path, gt_path):
    print("=" * 60)
    print("Loading data...")
    print(f"Prediction file: {pred_path}")
    print(f"Ground truth file: {gt_path}")

    pred_df = pd.read_csv(pred_path, index_col=0)
    gt_df = pd.read_csv(gt_path, index_col=0)

    print(f"Prediction data shape: {pred_df.shape}")
    print(f"Ground truth data shape: {gt_df.shape}")

    pred = pred_df.values
    gt = gt_df.values


    overall_pcc = calculate_overall_pcc(pred, gt)
    print(f"\nPCC : {overall_pcc:.6f}")

    rmse = calculate_rmse(pred, gt)
    print(f"RMSE: {rmse:.6f}")


    results = {
        'overall_pcc': overall_pcc,
        'overall_rmse': rmse
    }
    return results


if __name__ == "__main__":
    import os
    data_name = "dataset-ML"
    pred_path = f"C:/SpaDiT/Version1.0/result/{data_name}/SpaDiT_prediction.csv"
    gt_path = f"C:/SpaDiT/Version1.0/result/{data_name}/original.csv"

    if not os.path.exists(pred_path):
        print(f"Error: prediction file not found - {pred_path}")
    elif not os.path.exists(gt_path):
        print(f"Error: ground truth file not found - {gt_path}")
    else:
        results = evaluate_predictions(pred_path, gt_path)
