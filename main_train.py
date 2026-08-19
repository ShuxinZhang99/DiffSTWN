import numpy as np
import os, time, torch
from Model.Diffusion_Model import *
from utils.utils import GetLaplacian
from utils.earlystopping import EarlyStopping
from utils.ZeroInflated_NB_Loss import stable_zinb_nll_loss
from Data.dataset import TrafficDataset, CleanDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

epoch_num = 1000
lr = 0.0005
time_interval = 60      # time granularity
tg_in_one_day = 17      # time of a day
batch_size = 8
station_num = 41        # number of stations
od_pair_num = 41 * 41   # number of od pairs
ext_num = 5             # number of external factors
time_step = 238         # total time step
T_h = 17                # history_step
T_p = 8                # forecast_horizon

# ST-WaveNet
d_model = 512
diffusion_steps = 100   # forward diffusion steps
sample_steps = 100      # reverse denoising steps
sample_strategy = 'ddpm'  # ddim_one, ddpm
beta_schedule = 'quad'  # uniform
mask_ratio = 0

# saved_path
model_type = "ODMetro-60min-DiffSTWN"
TIMESTAMP = str(time.strftime("%Y_%m_%d_%H_%M_%S"))
save_dir = './save_model/' + model_type + '_' + TIMESTAMP
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# data_split
feature_file = '/root/autodl-tmp/DiffSTWN_code/Data/estimated_od/OD_estimated.csv'
real_file    = '/root/autodl-tmp/DiffSTWN_code/Data/real_od/OD_real.csv'
external_file = '/root/autodl-tmp/DiffSTWN_code/Data/external_data/external_infor.csv'
val_start_idx = 118 
test_start_idx = 203

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
adjacency_origin = np.eye(station_num)
adjacency_destination = np.eye(station_num)
adjacency_origin = torch.tensor(GetLaplacian(adjacency_origin).get_normalized_adj(station_num)).type(
    torch.float32).to(device)
adjacency_destination = torch.tensor(GetLaplacian(adjacency_destination).get_normalized_adj(station_num)).type(
    torch.float32).to(device)

# model setup
model = DiffWave(diffusion_steps, sample_steps, sample_strategy, adjacency_origin, adjacency_destination, d_model,
                 beta_schedule, fore_horizon=T_p, con_time_length=T_h + T_p, con_pre_len=T_h+T_p, num_nodes=od_pair_num,
                 num_stations=station_num, num_ext=ext_num, device=device).to(device)

# optimizer
mse = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopping = EarlyStopping(patience=100, verbose=True)

# training
for epoch in range(0, epoch_num):
    # model train
    train_loss = 0
    n = 0
    model.train()
    for i, batch in enumerate(train_loader):
        future, history, pos_w, pos_d, future_raw, ext_history, ext_future = batch
        x = torch.cat((history, future), dim=1).to(device)  # (B, T, V, F)
        # get x0_masked
        mask = torch.randint_like(history, low=0, high=100) < int(mask_ratio * 100)  # mask the history in a ratio with mask_ratio
        history[mask] = 0
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1)       # (B, T, V, F)
        ext_infor = torch.cat((ext_history, ext_future), dim=1)                # (B, T, V, F)

        x = x.transpose(1, 3).to(device)                  # (B, F, V, T)
        x_masked = x_masked.transpose(1, 3).to(device)    # (B, F, V, T)
        future_raw = future_raw.transpose(1, 3).to(device)            # (B, F, V, T/2)
        ext_infor = ext_infor.transpose(1, 3).to(device)  # (B, F, V, T)

        loss = model.loss(x, ext_infor, x_masked, future_raw)
        n += 1
        train_loss = train_loss * (n - 1) / n + loss.item() / n

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    with torch.no_grad():
        # model validation
        model.eval()
        val_loss = 0
        for i, batch in enumerate(val_loader):
            future, history, pos_w, pos_d, future_raw, ext_history, ext_future = batch
            x = torch.cat((history, future), dim=1).to(device)  # (B, T, V, F)
            # get x0_masked
            mask = torch.randint_like(history, low=0, high=100) < int(mask_ratio * 100)  # mask the history in a ratio with mask_ratio
            history[mask] = 0
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(device)  # (B, T, V, F)
            ext_infor = torch.cat((ext_history, ext_future), dim=1).to(device)  # (B, T, V, F)

            x = x.transpose(1, 3).to(device)                  # (B, F, V, T)
            x_masked = x_masked.transpose(1, 3)               # (B, F, V, T)
            future = future.transpose(1, 3).to(device)                    # (B, F ,V, T/2)
            future_raw = future_raw.transpose(1, 3).to(device)            # (B, F ,V, T/2)
            ext_infor = ext_infor.transpose(1, 3).to(device)  # (B, F, V, T)

            n_samples = 10
            x_hat, n_params, p, pi = model(x_masked, ext_infor, n_samples)  # x_hat (B, n_samples, V, T)  经过denosising后的输出，即x0
            x_hat = torch.clip(x_hat, 0)  # (B, n_samples, F, V, T)
            target = torch.mean(x_hat, dim=1)
            target = target.squeeze(1)
            target_value = target[:, :, -T_p:]

            loss_1 = mse(input=future.squeeze(1), target=target_value)
            loss_2 = stable_zinb_nll_loss(future_raw, n_params, p, pi)
            val_loss = 0.4 * loss_1.item() + 0.6 * loss_2.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print('epoch:', epoch, 'train Loss:', avg_train_loss, 'val Loss:', avg_val_loss)

    # training strategy
    if epoch > 0:
        model_dict = model.state_dict()
        # for k, v in model_dict.items():
        #     print(k)  # 只打印key值，不打印具体参数。
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopping(avg_val_loss, model_dict, model, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

