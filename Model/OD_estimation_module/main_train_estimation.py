import numpy as np
import os, time, torch
from torch.utils.tensorboard import SummaryWriter

from utils.earlystopping import EarlyStopping
from data.OD_dataset import *
from data.OD_get_dataloader import *
from real_time_OD_estimation import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
epoch_num = 200
lr = 0.0005
time_interval = 60
time_lag = 34
tg_in_one_day = 34
forecast_day_number = 2
pre_len = 34
batch_size = 8
station_num = 41 #62  #41
od_pair_num = 1681 #3844  #1681
T_h = 34
T_p = 34

model_type = "long_19_20OD_Estimated_30min"
TIMESTAMP = str(time.strftime("%Y_%m_%d_%H_%M_%S"))
save_dir = './save_model/' + model_type + '_' + TIMESTAMP
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


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

temp_time = time.time()
early_stopping = EarlyStopping(patience=100, verbose=True)


for epoch in range(0, epoch_num):
    # model train
    train_loss = 0
    model.train()
    for i, batch in enumerate(train_loader):
        unfinished_flow = batch['unfinished_flow'].type(torch.float32).to(device)
        long_term_rate = batch['long_term_rate'].type(torch.float32).to(device)
        short_term_rate = batch['short_term_rate'].type(torch.float32).to(device)
        finished_od = batch['finished_od'].type(torch.float32).to(device)
        future_od = batch['future_od'].type(torch.float32).to(device)
        complete_od = batch['complete_od'].type(torch.float32).to(device)

        estimated_od = model(unfinished_flow, long_term_rate, short_term_rate, finished_od)

        loss = mse(estimated_od, complete_od)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    with torch.no_grad():
        # model validation
        model.eval()
        val_loss = 0
        for i, batch in enumerate(val_loader):
            val_unfinished_flow = batch['unfinished_flow'].type(torch.float32).to(device)
            val_long_term_rate = batch['long_term_rate'].type(torch.float32).to(device)
            val_short_term_rate = batch['short_term_rate'].type(torch.float32).to(device)
            val_finished_od = batch['finished_od'].type(torch.float32).to(device)
            val_future_od = batch['future_od'].type(torch.float32).to(device)
            val_completed_od = batch['complete_od'].type(torch.float32).to(device)

            estimated_od = model(val_unfinished_flow, val_long_term_rate, val_short_term_rate, val_finished_od)

            loss = mse(estimated_od, val_completed_od)

            val_loss += loss.item() 
        

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    # scheduler.step(avg_val_loss)
    writer.add_scalar("loss_train", avg_train_loss, epoch)
    writer.add_scalar("loss_eval", avg_val_loss, epoch)
    print('epoch:', epoch, 'train Loss:', avg_train_loss, 'val Loss:', avg_val_loss)

    if epoch > 0:
        model_dict = model.state_dict()
        # for k, v in model_dict.items():
        #     print(k)  # 只打印key值，不打印具体参数。
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopping(avg_val_loss, model_dict, model, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

    # 每10个epoch打印一次训练时间
    if epoch % 10 == 0:
        print("time for 10 epoches:", round(time.time() - temp_time, 2))
        temp_time = time.time()
global_end_time = time.time() - global_strat_time
print("global end time:", global_end_time)



