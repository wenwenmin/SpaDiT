import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from preprocess.utils import calculate_rmse_per_gene, calculate_pcc_per_gene,calculate_pcc_with_mask,calculate_rmse_with_mask
from preprocess.utils import mask_tensor_with_masks
def model_sample_diff(model, device, dataloader, total_sample, time, is_condi, condi_flag):
    noise = []
    i = 0
    for _, x_hat, x_cond in dataloader: # 计算整个shape得噪声 一次循环算batch大小  加上了celltype 去掉了, celltype
        x_hat, x_cond = x_hat.float().to(device), x_cond.float().to(device) # x.float().to(device)
        t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)
        # celltype = celltype.to(device)
        if not is_condi:
            n = model(total_sample[i:i+len(x_cond)], t, None) # 一次计算batch大小得噪声
        else:
            n = model(total_sample[i:i+len(x_cond)], x_hat, t, x_cond, condi_flag=condi_flag) # 加上了celltype 去掉了, celltype
        noise.append(n)
        i = i+len(x_cond)
    noise = torch.cat(noise, dim=0)
    return noise

def sample_diff(model,
                dataloader,
                noise_scheduler,
                mask_nonzero_ratio = None,
                mask_zero_ratio = None,
                gt = None,
                sc = None,
                device=torch.device('cuda:0'),
                num_step=1000,
                sample_shape=(7060, 2000),
                is_condi=False,
                sample_intermediate=200,
                model_pred_type: str = 'noise',
                is_classifier_guidance=False,
                omega=0.1,
                is_tqdm = True):
    model.eval()
    gt = torch.tensor(gt).to(device)
    sc = torch.tensor(sc).to(device)
    x_t = torch.randn(sample_shape[0], sample_shape[1]).to(device)
    timesteps = list(range(num_step))[::-1]  # 倒序
    gt_mask, mask_nonzero, mask_zero = mask_tensor_with_masks(gt, mask_zero_ratio, mask_nonzero_ratio)
    mask = torch.tensor(mask_nonzero).to(device)
    # mask = None
    # x_t =  x_t * (1 - mask) + gt * mask
    # x_t = x_t  + gt * mask
    x_t = x_t
    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]

    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            # 输出噪声
            model_output = model_sample_diff(model,
                                        device=device,
                                        dataloader=dataloader,
                                        total_sample=x_t,  # x_t
                                        time=time,  # t
                                        is_condi=is_condi,
                                        condi_flag=True)
            if is_classifier_guidance:
                model_output_uncondi = model_sample_diff(model,
                                                    device=device,
                                                    dataloader=dataloader,
                                                    total_sample=x_t,
                                                    time=time,
                                                    is_condi=is_condi,
                                                    condi_flag=False)
                model_output = (1 + omega) * model_output - omega * model_output_uncondi

        # 计算x_{t-1}
        x_t, _ = noise_scheduler.step(model_output,  # 一般是噪声
                                     torch.from_numpy(np.array(time)).long().to(device),
                                      x_t,
                                         model_pred_type=model_pred_type)
        # epoch_pcc = calculate_pcc_with_mask(x_t, gt_mask, mask_nonzero)
        # epoch_rmse = calculate_rmse_with_mask(x_t, gt_mask, mask_nonzero)
        epoch_pcc = calculate_pcc_per_gene(x_t, gt)
        epoch_rmse = calculate_rmse_per_gene(x_t, gt)
        ts.set_postfix_str(f'PCC:{epoch_pcc:.5f}, RMSE:'
                           f'{epoch_rmse:.5f}')
        if mask is not None:
            x_t = x_t *  mask + (1 - mask) * gt

        if time == 0 and model_pred_type == 'x_start':
            # 如果直接预测 x_0 的话，最后一步直接输出
            sample = model_output


    recon_x = x_t.detach().cpu().numpy()
    return recon_x

