import numpy as np
import pandas as pd
import os, time, torch
from TRB.DiffSTWN_code.utils.utils import GetLaplacian
from TRB.DiffSTWN_code.utils.metrics import *
from Model.Diffusion_Model import *
from Data.dataset import TrafficDataset, CleanDataset

device = torch.device("cuda:0")  # if torch.cuda.is_available() else "cpu")
print(device)

epoch_num = 1000
lr = 0.0005
time_interval = 60      # time granularity
tg_in_one_day = 18      # time of a day
batch_size = 8
station_num = 61        # number of stations
od_pair_num = 61 * 61   # number of od pairs
ext_num = 5             # number of external factors
time_step = 756         # total time step
T_h = 18                # history_step
T_p = 18                # forecast_horizon

# ST-WaveNet
d_model = 512
diffusion_steps = 100   # forward diffusion steps
sample_steps = 100      # reverse denoising steps
sample_strategy = 'ddpm'  # ddim_one, ddpm
beta_schedule = 'quad'  # uniform
mask_ratio = 0

# data_split
feature_file = './DiffSTWN/Data/estimated_od/estimated_OD_demand_data.npy'
real_file    = './DiffSTWN/Data/real_od/real_OD_demand_data.npy'
external_file = './DiffSTWN/data/external_data/external_factors_data.npy'
val_start_idx = 503 #1942  # 867 int(1190 * 0.6)
test_start_idx = 629   #2213  # 1020  # int(1190 * 0.8)


clean_data = CleanDataset(feature_file=feature_file, real_file=real_file, val_start_idx=val_start_idx,
                          external_file=external_file, normalization_type='minmax')
# train_loader
train_dataset = TrafficDataset(clean_data, (0 + T_p, val_start_idx - T_p), T_h=T_h, T_p=T_p, V=od_pair_num,
                               points_per_hour=1)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False, pin_memory=True)
# val_loader
val_dataset = TrafficDataset(clean_data, (val_start_idx + T_p, test_start_idx - T_p), T_h=T_h, T_p=T_p, V=od_pair_num,
                             points_per_hour=1)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
# test_loader
test_dataset = TrafficDataset(clean_data, (test_start_idx + T_p, time_step - T_p), T_h=T_h, T_p=T_p, V=od_pair_num,  # 2298
                              points_per_hour=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)


# get multi_graph (instance)
adjacency_origin = np.eye(od_pair_num)
adjacency_destination = np.eye(od_pair_num)
adjacency_origin = torch.tensor(GetLaplacian(adjacency_origin).get_normalized_adj(od_pair_num)).type(
    torch.float32).to(device)
adjacency_destination = torch.tensor(GetLaplacian(adjacency_destination).get_normalized_adj(od_pair_num)).type(
    torch.float32).to(device)


# 分批生成扩散样本
@torch.no_grad()
def generate_diffusion_samples_in_chunks(
    model,
    x_masked,
    ext_inputs,
    total_samples,
    sample_chunk_size,
    pre_len,
    clean_data,
):
    """
    分批调用 DiffWave，避免一次在 GPU 上生成大量样本。

    Parameters
    ----------
    total_samples:
        最终需要的总样本数，例如 50。

    sample_chunk_size:
        每次在 GPU 上生成的样本数，例如 2 或 5。

    Returns
    -------
    samples:
        [B,total_samples,V,pre_len]，CPU Tensor，
        已经反归一化到原始客流尺度。
    """

    sample_chunks = []
    generated_samples = 0

    while generated_samples < total_samples:

        current_num = min(
            sample_chunk_size,
            total_samples - generated_samples,
        )

        # x_hat:
        # [B,current_num,F,V,T_h+T_p]
        x_hat, n_params, p, pi = model(
            x_masked,
            ext_inputs,
            current_num,
        )

        if x_hat.ndim != 5:
            raise ValueError(
                "DiffWave 输出应为 [B,S,F,V,T]，"
                f"当前形状为 {tuple(x_hat.shape)}"
            )

        # 只取未来部分
        x_hat_future = x_hat[
            ...,
            -pre_len:
        ]

        # 反归一化，并移到CPU
        x_hat_future = inverse_normalize_to_cpu(
            clean_data,
            x_hat_future,
        )

        # 客流预测负值截断为0
        x_hat_future = torch.nan_to_num(
            x_hat_future,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        x_hat_future = torch.clamp(
            x_hat_future,
            min=0.0,
        )

        # [B,S,1,V,T] -> [B,S,V,T]
        if x_hat_future.shape[2] == 1:
            x_hat_future = x_hat_future.squeeze(2)
        else:
            raise ValueError(
                "当前代码假设客流特征数 F=1，"
                f"但得到 F={x_hat_future.shape[2]}"
            )

        sample_chunks.append(x_hat_future)

        generated_samples += current_num

        print(
            f"\rGenerated diffusion samples: "
            f"{generated_samples}/{total_samples}",
            end="",
            flush=True,
        )

        del x_hat
        del x_hat_future

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()

    samples = torch.cat(
        sample_chunks,
        dim=1,
    )

    return samples

global_strat_time = time.time()

# model setup
model = DiffWave(diffusion_steps, sample_steps, sample_strategy, adjacency_origin, adjacency_destination, d_model,
                 beta_schedule, con_time_length=T_h + T_p, con_pre_len=T_h+T_p, num_nodes=od_pair_num,
                 num_stations=station_num, num_ext=ext_num).to(device)

mse = torch.nn.MSELoss().to(device)

# model loading
path = './save_model/ODMetro-60min-DiffSTWN/model_dict_checkpoint_773_0.00001803.pth'
checkpoint = torch.load(path, map_location=device, weights_only=False)

# 判断保存的是什么类型
print("checkpoint type:", type(checkpoint))

if isinstance(checkpoint, torch.nn.Module):
    # checkpoint本身就是完整模型
    model = checkpoint.to(device)

elif isinstance(checkpoint, dict):
    print("checkpoint keys:", checkpoint.keys())

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

    elif "state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["state_dict"],
            strict=True
        )

    elif "model" in checkpoint:
        if isinstance(checkpoint["model"], torch.nn.Module):
            model = checkpoint["model"].to(device)
        else:
            model.load_state_dict(
                checkpoint["model"],
                strict=True
            )

    else:
        # 假设整个字典就是state_dict
        model.load_state_dict(checkpoint, strict=True)

else:
    raise TypeError(
        f"无法识别checkpoint类型：{type(checkpoint)}"
    )

# test
total_diffusion_samples = 80
sample_chunk_size = 2

# 统计量
sum_squared_error = 0.0
sum_absolute_error = 0.0
sum_absolute_truth = 0.0
total_point_count = 0

coverage_count = 0
interval_width_sum = 0.0

crps_sum = 0.0
crps_point_count = 0

with torch.no_grad():
    model.eval()
    test_loss = 0
    samples, actual = [], []
    for i, batch in enumerate(test_loader):
        future, history, pos_w, pos_d, future_raw, ext_history, ext_future = batch
        future = future.float()
        history = history.float()
        masked_history = history.clone()

        x = torch.cat((history, future), dim=1).to(device)  # (B, T, V, F)
        
        mask = torch.randint_like(history, low=0, high=100) < int(mask_ratio * 100)  # mask the history in a ratio with mask_ratio
        masked_history[mask] = 0
        x_masked = torch.cat((masked_history, torch.zeros_like(future)), dim=1).to(device)  # (B, T, V, F)
        ext_infor = torch.cat((ext_history, ext_future), dim=1).to(device)                  # (B, T, V, F)

        x = x.transpose(1, 3)  # (B, F, V, T)
        x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)
        future = future.transpose(1, 3).to(device)  # (B, F ,V, T/2)
        future_raw = future_raw.transpose(1, 3).to(device)
        future_true = future_raw.squeeze(1).float().cpu()
        ext_infor = ext_infor.transpose(1, 3).to(device)  # (B, F, V, T)

        future_true = torch.nan_to_num(
            future_true,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        future_true = torch.clamp(
            future_true,
            min=0.0,
        )

        # 分批生成扩散样本
        future_samples = (
            generate_diffusion_samples_in_chunks(
                model=model,
                x_masked=x_masked,
                ext_inputs=ext_infor,
                total_samples=(
                    total_diffusion_samples
                ),
                sample_chunk_size=(
                    sample_chunk_size
                ),
                pre_len=T_p,
                clean_data=clean_data,
            )
        )

        # 计算样本均值、区间和CRPS
        (
            prediction_mean,
            lower95,
            upper95,
            crps_values,
        ) = calculate_sample_distribution_metrics(
            samples=future_samples,
            target=future_true,
        )

        # RMSE / MAE / WMAPE
        error = (
            prediction_mean
            - future_true
        )

        sum_squared_error += torch.sum(
            error.double() ** 2
        ).item()

        sum_absolute_error += torch.sum(
            torch.abs(error).double()
        ).item()

        sum_absolute_truth += torch.sum(
            torch.abs(future_true).double()
        ).item()

        total_point_count += future_true.numel()

        # PICP-95
        covered = (
            (future_true >= lower95)
            & (future_true <= upper95)
        )

        coverage_count += covered.sum().item()

        # MPIW-95
        interval_width_sum += torch.sum(
            (upper95 - lower95).double()
        ).item()

        # CRPS
        crps_sum += torch.sum(
            crps_values.double()
        ).item()

        crps_point_count += crps_values.numel()

        del x_masked
        del history
        del future_samples
        del future_true
        del prediction_mean
        del lower95
        del upper95
        del crps_values

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 汇总最终指标
eps = 1e-8

RMSE = np.sqrt(
    sum_squared_error
    / max(total_point_count, 1)
)

MAE = (
    sum_absolute_error
    / max(total_point_count, 1)
)

# 输出百分数，例如 12.5 表示 12.5%
WMAPE = (
    sum_absolute_error
    / max(sum_absolute_truth, eps)
    * 100.0
)

PICP95 = (
    coverage_count
    / max(total_point_count, 1)
)

MPIW95 = (
    interval_width_sum
    / max(total_point_count, 1)
)

CRPS = (
    crps_sum
    / max(crps_point_count, 1)
)


print("=" * 60)
print("DiffWave test results")
print("=" * 60)

print(f"RMSE:    {RMSE:.6f}")
print(f"MAE:     {MAE:.6f}")
print(f"WMAPE:   {WMAPE:.6f}%")
print(f"CRPS:    {CRPS:.6f}")
print(f"PICP-95: {PICP95:.6f}")
print(f"MPIW-95: {MPIW95:.6f}")
