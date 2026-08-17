import numpy as np
import os, time, torch
from torch.utils.tensorboard import SummaryWriter

from utils.earlystopping import EarlyStopping
from utils.metrics import Metrics
from data.OD_dataset import *
from data.OD_get_dataloader import *
from real_time_OD_estimation import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
epoch_num = 1000
lr = 0.0005
time_interval = 60
time_lag = 17
tg_in_one_day = 17
forecast_day_number = 2
pre_len = 17
batch_size = 8
station_num = 41
od_pair_num = 1681
T_h = 17
T_p = 17


train_loader, val_loader, test_loader, count_scale = get_realtime_od_forecast_dataloader(
     T_h=T_h,
     T_p=T_p,
     T_in_a_day=tg_in_one_day,
     forecast_day_number=forecast_day_number,
     batch_size=batch_size
)


global_strat_time = time.time()
writer = SummaryWriter()

model = RealTimeODEstimator(time_lag=time_lag, num_stations=station_num, hidden_dim=8, kernel_size=3, device=device)
print(model)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = torch.nn.MSELoss().to(device)

path = '/root/autodl-tmp/DiffSTWN/Model/OD_estimation_module/save_model/19_20OD_Estimated_60min_17_2026_07_01_00_41_04/model_dict_checkpoint_195_0.00000487.pth'
checkpoint = torch.load(path)
model.load_state_dict(checkpoint, strict=True)


# test
result = []
result_original = []
result_finished = []

if not os.path.exists('result/prediction'):
    os.makedirs('result/prediction/')
if not os.path.exists('result/original'):
    os.makedirs('result/original/')
with torch.no_grad():
    model.eval()
    test_loss = 0
    for i, batch in enumerate(test_loader):
        unfinished_flow = batch['unfinished_flow'].type(torch.float32).to(device)
        long_term_rate = batch['long_term_rate'].type(torch.float32).to(device)
        short_term_rate = batch['short_term_rate'].type(torch.float32).to(device)
        finished_od = batch['finished_od'].type(torch.float32).to(device)
        future_od = batch['future_od'].type(torch.float32).to(device)
        complete_od = batch['complete_od'].type(torch.float32).to(device)

        estimated_od = model(unfinished_flow, long_term_rate, short_term_rate, finished_od)
        loss = mse(estimated_od, complete_od)

        test_loss += loss.item()

        # evaluate on original scale
        # 获取result (batch, 276, pre_len)
        clone_prediction = estimated_od * count_scale # clone(): Copy the tensor and allocate the new memory
        clone_prediction = clone_prediction.cpu().detach().numpy()
        # print(clone_prediction.shape)  # (16, 276, 1)
        for i in range(clone_prediction.shape[0]):
            result.append(clone_prediction[i])

        # 获取result_original
        clone_real = complete_od * count_scale
        clone_real = clone_real.cpu().detach().numpy()
        # print(test_inflow_Y_original.shape)  # (16, 276, 1)
        for i in range(clone_real.shape[0]):
            result_original.append(clone_real[i])

        clone_finished = finished_od * count_scale
        clone_finished = clone_finished.cpu().detach().numpy()
        for i in range(clone_finished.shape[0]):
            result_finished.append(clone_finished[i])

    # 计算RMSE\MAE\WMAPE
    print(np.array(result).shape, np.array(result_original).shape, np.array(result_finished).shape)
    # 取整&负数取0
    result = np.array(result).astype(int)
    result[result < 0] = 0
    result_original = np.array(result_original).astype(int)
    result_original[result_original < 0] = 0
    result_finished = np.array(result_finished).astype(int)
    result_finished[result_finished < 0] = 0

    result = result[-1].reshape(od_pair_num, -1)  # station * station
    result_original = result_original[-1].reshape(od_pair_num, -1)
    result_finished = result_finished[-1].reshape(od_pair_num, -1)

    print(result.shape, result_original.shape, result_finished.shape)

    
    pd.DataFrame(result).to_csv('/root/autodl-tmp/DiffSTWN/model/my_model/result/1920OD_60MIN_estimated.csv')
    pd.DataFrame(result_original).to_csv('/root/autodl-tmp/DiffSTWN/model/my_model/result/1920OD_60MIN_real.csv')
    pd.DataFrame(result_finished).to_csv('/root/autodl-tmp/DiffSTWN/model/my_model/result/1920OD_60MIN_finished.csv')

    RMSE, R2, MAE, WMAPE = Metrics(result_original, result).evaluate_performance()

    RMSE_2, R2_2, MAE_2, WMAPE_2 = Metrics(result_original, result_finished).evaluate_performance()

    avg_test_loss = test_loss / len(test_loader)
    print('test Loss:', avg_test_loss)





