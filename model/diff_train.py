import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat

import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import sys
import os
from torch.optim.lr_scheduler import StepLR

from .diff_scheduler import NoiseScheduler
from preprocess.utils import mask_tensor_with_masks
import torch.nn.functional as F



class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        abs_error = torch.abs(input - target)
        is_small_error = abs_error <= self.delta
        small_error_loss = 0.5 * abs_error ** 2
        large_error_loss = self.delta * abs_error - 0.5 * self.delta ** 2
        return torch.where(is_small_error, small_error_loss, large_error_loss).mean()

class diffusion_loss(nn.Module):
    def __init__(self, penalty_factor=1.0, delta=1.0):
        super(diffusion_loss, self).__init__()
        self.mse = nn.MSELoss()
        self.huber_loss = HuberLoss(delta=delta)
        self.penalty_factor = penalty_factor

    def forward(self, y_pred_0, y_true_0, y_pred_1, y_true_1):
        loss_mse = self.mse(y_pred_0, y_true_0)
        loss_huber = self.huber_loss(y_pred_1, y_true_1) * self.penalty_factor  # 计算 Huber 损失
        return loss_mse + loss_huber




def normal_train_diff(model,
                 dataloader,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 pred_type: str = 'noise',
                 diffusion_step: int = 1000,
                 device=torch.device('cuda:0'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 mask_nonzero_ratio= None,
                 mask_zero_ratio = None):
    """通用训练函数

    Args:
        lr (float):
        momentum (float): 动量
        max_iteration (int, optional): 训练的 iteration. Defaults to 30000.
        pred_type (str, optional): 预测的类型噪声或者 x_0. Defaults to 'noise'.
        batch_size (int, optional):  Defaults to 1024.
        diffusion_step (int, optional): 扩散步数. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:0').
        is_class_condi (bool, optional): 是否采用condition. Defaults to False.
        is_tqdm (bool, optional): 开启进度条. Defaults to True.
        is_tune (bool, optional): 是否用 ray tune. Defaults to False.
        condi_drop_rate (float, optional): 是否采用 classifier free guidance 设置 drop rate. Defaults to 0..

    Raises:
        NotImplementedError: _description_
    """

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    criterion = diffusion_loss()
    # criterion = nn.MSELoss()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_hat, x_cond) in enumerate(dataloader): # 去掉了, celltype
            x, x_hat, x_cond = x.float().to(device), x_hat.float().to(device),x_cond.float().to(device)
            # celltype = celltype.to(device)
            x, x_nonzero_mask, x_zero_mask = mask_tensor_with_masks(x, mask_zero_ratio, mask_nonzero_ratio)
            x_hat, x_hat_nonzero_mask, x_hat_zero_mask = mask_tensor_with_masks(x_hat, mask_zero_ratio, mask_nonzero_ratio)

            x_noise = torch.randn(x.shape).to(device)
            x_hat_noise = torch.randn(x_hat.shape).to(device)

            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()
            timesteps = timesteps.to(device)
            x_t = noise_scheduler.add_noise(x,
                                            x_noise,
                                            timesteps=timesteps)

            x_hat_t = noise_scheduler.add_noise(x_hat,
                                            x_hat_noise,
                                            timesteps=timesteps)

            # mask = torch.tensor(mask).to(device)
            # mask = (1-((torch.rand(x.shape[1]) < mask_ratio).int())).to(device)

            x_noisy = x_t * x_nonzero_mask + x * (1 - x_nonzero_mask)
            x_hat_noisy = x_hat_t * x_hat_nonzero_mask + x_hat * (1 - x_hat_nonzero_mask)

            noise_pred = model(x_noisy, x_hat_noisy, t=timesteps.to(device), y=x_cond) # 去掉了, z=celltype
            # loss = criterion(noise_pred, noise)

            loss = criterion(x_noise * x_nonzero_mask, noise_pred * x_nonzero_mask, x_noise * x_zero_mask,
                             noise_pred *  x_zero_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss = epoch_loss / (i + 1)  # type: ignore

        current_lr = optimizer.param_groups[0]['lr']

        # 更新tqdm的描述信息
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}, lr:{current_lr:.2e}')  # type: ignore

        if is_tune:
            session.report({'loss': epoch_loss})
