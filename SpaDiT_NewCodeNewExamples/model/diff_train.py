import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import math

from .diff_scheduler import NoiseScheduler
from preprocess.utils import mask_tensor_with_masks


class ReconstructionLoss(nn.Module):
    def __init__(self, vel_weight=0.5, nonzero_weight=2.0, zero_weight=0.1):
        super(ReconstructionLoss, self).__init__()
        self.vel_weight = vel_weight
        self.nonzero_weight = nonzero_weight
        self.zero_weight = zero_weight

    def forward(self, v_pred, v_true, x_pred, x_true):
        vel_loss = ((v_pred - v_true) ** 2).mean()

        recon_loss = ((x_pred - x_true) ** 2).mean()

        loss = self.vel_weight * vel_loss + (1 - self.vel_weight) * recon_loss

        return loss, vel_loss, recon_loss


class SparseWeightedLoss(nn.Module):
    def __init__(self, vel_weight=0.3, nonzero_weight=3.0, zero_weight=0.5):
        super(SparseWeightedLoss, self).__init__()
        self.vel_weight = vel_weight
        self.nonzero_weight = nonzero_weight
        self.zero_weight = zero_weight

    def forward(self, v_pred, v_true, x_pred, x_true):
        nonzero_mask = (x_true != 0).float()
        weights = nonzero_mask * (self.nonzero_weight - 1.0) + 1.0

        vel_diff = (v_pred - v_true) ** 2
        vel_loss = (vel_diff * weights).sum() / (weights.sum() + 1e-8)

        recon_diff = (x_pred - x_true) ** 2
        recon_loss = (recon_diff * weights).sum() / (weights.sum() + 1e-8)

        loss = self.vel_weight * vel_loss + (1 - self.vel_weight) * recon_loss

        return loss, vel_loss, recon_loss


def normal_train_diff(model,
                 dataloader,
                 lr: float = 1e-4,
                 num_epoch: int = 100,
                 pred_type: str = 'x_pred',
                 diffusion_step: int = 1000,
                 device=torch.device('cuda:0'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 mask_nonzero_ratio= None,
                 mask_zero_ratio = None,
                 warmup_epochs: int = 10,
                 min_lr_ratio: float = 0.01):
    criterion = SparseWeightedLoss(
        vel_weight=0.3,
        nonzero_weight=3.0,
        zero_weight=0.5
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=lr * min_lr_ratio
    )

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_hat, x_cond) in enumerate(dataloader):
            x, x_hat, x_cond = x.float().to(device), x_hat.float().to(device), x_cond.float().to(device)

            x_noise = torch.randn(x.shape).to(device)
            x_hat_noise = torch.randn(x_hat.shape).to(device)

            timesteps = torch.rand(x.shape[0]).to(device) * 0.90 + 0.05
            t_normalized = timesteps

            z = t_normalized.unsqueeze(1) * x + (1 - t_normalized).unsqueeze(1) * x_noise
            z_hat = t_normalized.unsqueeze(1) * x_hat + (1 - t_normalized).unsqueeze(1) * x_hat_noise

            denominator = (1 - t_normalized).unsqueeze(1)
            denominator = torch.clamp_min(denominator, 0.05)
            v = (x - z) / denominator

            x_pred = model(z, z_hat, t=timesteps, y=x_cond)

            v_pred = (x_pred - z) / denominator

            if (mask_nonzero_ratio is not None and mask_zero_ratio is not None
                and (mask_nonzero_ratio > 0 or mask_zero_ratio > 0)):
                x, x_nonzero_mask, x_zero_mask = mask_tensor_with_masks(x, mask_zero_ratio, mask_nonzero_ratio)
                loss, vel_loss, recon_loss = criterion(v_pred * x_nonzero_mask, v * x_nonzero_mask,
                                                       x_pred * x_nonzero_mask, x * x_nonzero_mask)
            else:
                loss, vel_loss, recon_loss = criterion(v_pred, v, x_pred, x)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss = epoch_loss / (i + 1)

        current_lr = optimizer.param_groups[0]['lr']

        if is_tqdm:
            t_epoch.set_postfix_str(f'loss:{epoch_loss:.5f}, vel:{vel_loss:.5f}, recon:{recon_loss:.5f}, lr:{current_lr:.2e}')

        if is_tune:
            from ray.air import session
            session.report({'loss': epoch_loss})
