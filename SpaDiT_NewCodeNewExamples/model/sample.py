import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from preprocess.utils import calculate_rmse_per_gene, calculate_pcc_per_gene,calculate_pcc_with_mask,calculate_rmse_with_mask
from preprocess.utils import mask_tensor_with_masks


def model_sample_diff(model, device, dataloader, x_t, time, is_condi, sc_data):
    noise = []
    for _, x_hat_batch, x_cond in dataloader:
        x_hat_batch = x_hat_batch.float().to(device)
        x_cond = x_cond.float().to(device)
        batch_size = x_hat_batch.shape[0]
        t = torch.full((batch_size,), time, dtype=torch.float32, device=device)

        x_batch = x_t[:batch_size].float().to(device)

        if not is_condi:
            n = model(x_batch, t, None)
        else:
            n = model(x_batch, x_hat_batch, t, x_cond)
        noise.append(n)
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
                model_pred_type: str = 'x_pred',
                is_classifier_guidance=False,
                omega=0.1,
                is_tqdm = True,
                mask_all_spots = False):
    model.eval()
    gt = torch.tensor(gt).to(device)
    sc = torch.tensor(sc).to(device)

    x_t = torch.randn(gt.shape[0], gt.shape[1]).to(device)

    timesteps = torch.linspace(1.0, 0.0, num_step + 1)

    if sample_intermediate and sample_intermediate < num_step:
        indices = torch.linspace(0, num_step - 1, sample_intermediate).long()
        timesteps = timesteps[indices]

    ts = tqdm(timesteps[:num_step])
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time:.4f}')
        next_time = timesteps[t_idx + 1] if t_idx + 1 < len(timesteps) else torch.tensor(0.0)

        with torch.no_grad():
            model_output = model_sample_diff(model,
                                        device=device,
                                        dataloader=dataloader,
                                        x_t=x_t,
                                        time=time.item(),
                                        is_condi=is_condi,
                                        sc_data=sc)

            if is_classifier_guidance:
                model_output_uncondi = model_sample_diff(model,
                                                    device=device,
                                                    dataloader=dataloader,
                                                    x_t=x_t,
                                                    time=time.item(),
                                                    is_condi=is_condi,
                                                    sc_data=sc)
                model_output = (1 + omega) * model_output - omega * model_output_uncondi

            x_t = noise_scheduler.velocity_sample_step(
                model_output=model_output,
                timestep=time.to(device),
                sample=x_t,
                next_timestep=next_time.to(device)
            )

        epoch_pcc = calculate_pcc_per_gene(x_t, gt)
        epoch_rmse = calculate_rmse_per_gene(x_t, gt)
        ts.set_postfix_str(f'PCC:{epoch_pcc:.5f}, RMSE:{epoch_rmse:.5f}')

        if mask_all_spots:
            mask = torch.ones_like(gt)
            x_t = x_t * mask + (1 - mask) * gt
        elif mask_nonzero_ratio is not None and mask_zero_ratio is not None:
            gt_mask, mask_nonzero, mask_zero = mask_tensor_with_masks(gt, mask_zero_ratio, mask_nonzero_ratio)
            mask = torch.tensor(mask_nonzero).to(device)
            x_t = x_t * mask + (1 - mask) * gt

    recon_x = x_t.detach().cpu().numpy()
    return recon_x
