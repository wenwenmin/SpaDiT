import anndata as ad
import numpy as np
import pandas as pd
import sys
import pickle
import os
import datetime
import time as tm
from functools import partial
import scipy.stats as st
from scipy.stats import wasserstein_distance
import scipy.stats
import copy
from sklearn.model_selection import KFold
import pandas as pd
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats
import seaborn as sns
import torch
from scipy.spatial.distance import cdist
import h5py
import time
import sys
import tangram as tg
import pickle
import yaml
import argparse
from os.path import join
from IPython.display import display
from model.diff_model import DiT_diff
from model.diff_scheduler import NoiseScheduler
from model.diff_train import normal_train_diff
from model.sample import sample_diff
from preprocess.result_analysis import clustering_metrics
from preprocess.utils import *
from preprocess.data import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sc_data", type=str, default='_sc.h5ad')
parser.add_argument("--st_data", type=str, default='_st.h5ad')


parser.add_argument("--document", type=str, default='dataset-ML')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--diffusion_step", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--depth", type=int, default=16)
parser.add_argument("--noise_std", type=float, default=10)
parser.add_argument("--pca_dim", type=int, default=100)
parser.add_argument("--head", type=int, default=16)
parser.add_argument("--mask_nonzero_ratio", type=float, default=0.3)
parser.add_argument("--mask_zero_ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=3407)
args = parser.parse_args()

print(os.getcwd())
print(torch.cuda.get_device_name(torch.cuda.current_device()))


def train_valid_test():
    seed_everything(args.seed)
    st_path = 'datasets/' + args.document + '/st/' + args.document + args.st_data
    sc_path = 'datasets/' + args.document + '/sc/' + args.document + args.sc_data
    gene_path = 'datasets/' + args.document + '/gene/'

    directory = 'save/' + args.document + '_ckpt/' + args.document + '_scdiff'

    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, args.document + '.pt')


    dataset = ConditionalDiffusionDataset(sc_path, st_path, gene_path,
                                          train_ratio=0.7, val_ratio=0.2,
                                          test_ratio=0.1, random_state=42)

    print("=" * 50)
    print("Dataset Tensor Dimensions:")
    print(f"  dataset.sc_data: {dataset.sc_data.shape}  (Cell count × Gene count)")
    print(f"  dataset.st_data: {dataset.st_data.shape}  (Spot count × Gene count)")
    print(f"  Number of shared genes: {len(dataset.gene_names)}")
    print("=" * 50)


    train_indices = dataset.get_train_gene_indices()
    val_indices = dataset.get_val_gene_indices()
    test_indices = dataset.get_test_gene_indices()


    train_mask = torch.zeros(len(dataset.gene_names), dtype=torch.bool)
    train_mask[train_indices] = True

    val_mask = torch.zeros(len(dataset.gene_names), dtype=torch.bool)
    val_mask[val_indices] = True

    test_mask = torch.zeros(len(dataset.gene_names), dtype=torch.bool)
    test_mask[test_indices] = True


    train_dataset = MaskedDataset(dataset, train_mask)
    valid_dataset = MaskedDataset(dataset, val_mask)
    test_dataset = MaskedDataset(dataset, test_mask)


    train_gene_names = [dataset.gene_names[i] for i in train_indices]
    val_gene_names = [dataset.gene_names[i] for i in val_indices]
    test_gene_names = [dataset.gene_names[i] for i in test_indices]

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_condition)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_condition)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_condition)

    cell_num = dataset.sc_data.shape[0]
    spot_num = dataset.st_data.shape[0]
    gene_num = len(dataset.gene_names)


    model = DiT_diff(
        st_input_size=spot_num,
        condi_input_size=cell_num,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.head,
        classes=6,
        mlp_ratio=4.0,
        pca_dim=args.pca_dim,
        dit_type='dit',
        gene_num=gene_num
    )

    model.to(args.device)
    diffusion_step = args.diffusion_step

    model.train()

    if not os.path.isfile(save_path):
        normal_train_diff(model,
                          dataloader=train_dataloader,
                          lr=args.learning_rate,
                          num_epoch=args.epoch,
                          diffusion_step=diffusion_step,
                          device=args.device,
                          pred_type='x_pred',
                          mask_nonzero_ratio=args.mask_nonzero_ratio,
                          mask_zero_ratio=args.mask_zero_ratio)
        torch.save(model.state_dict(), save_path)
    else:
        model.load_state_dict(torch.load(save_path))

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    model.eval()

    with torch.no_grad():
       test_gt = torch.stack([data for data, t, _ in test_dataset])
       test_sc = torch.stack([t for data, t, _ in test_dataset])
       prediction = sample_diff(model,
                                device=args.device,
                                dataloader=test_dataloader,
                                noise_scheduler=noise_scheduler,
                                mask_nonzero_ratio=0.95,
                                mask_zero_ratio=1,
                                mask_all_spots=False,
                                gt=test_gt,
                                sc=test_sc,
                                num_step=diffusion_step,
                                sample_shape=(test_gt.shape[0], test_gt.shape[1]),
                                is_condi=True,
                                sample_intermediate=diffusion_step,
                                model_pred_type='x_pred',
                                is_classifier_guidance=False,
                                omega=0.9
                                )

    return prediction, test_gt, test_gene_names


Data =  args.document
outdir = 'result/' + Data +'/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

hyper_directory = 'save/'+Data+'_ckpt/'+Data+'_hyper/'
hyper_file = Data + '_hyperameters.yaml'
hyper_full_path = os.path.join(hyper_directory, hyper_file)
if not os.path.exists(hyper_directory):
    os.makedirs(hyper_directory)
args_dict = vars(args)
with open(hyper_full_path, 'w') as yaml_file:
    yaml.dump(args_dict, yaml_file)

prediction_result, ground_truth, test_gene_names = train_valid_test()


gene_name = test_gene_names


prediction_result = prediction_result.T
ground_truth = ground_truth.numpy().T


pred_result = pd.DataFrame(prediction_result, columns=gene_name)
original = pd.DataFrame(ground_truth, columns=gene_name)
pred_result.to_csv(outdir + '/SpaDiT_prediction.csv', header=True, index=True)
original.to_csv(outdir + '/original.csv', header=True, index=True)


