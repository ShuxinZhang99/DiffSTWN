import torch
from torch.utils.data import DataLoader
from OD_dataset import RealTimeODIndexDataset


# =========================
# 根据你的实际路径修改
# =========================
unfinished_flow_file = "/Model/OD_estimated_module/data/unfinished_flow/un_inflow_60min_19_20.csv"
distribution_rate_file = "/Model/OD_estimated_module/data/distribution_rate/19_20NY_distrition_rate.npy"
finished_od_file = "/Model/OD_estimated_module/data/finished_od/19_20NY_finished_OD.npy"
# 真实完整 OD 需求，用于未来 T_p 步预测标签
future_od_file = "/Model/OD_estimated_module/data/real_od/19_20NY_full_OD.npy"

def get_realtime_od_forecast_dataloader(
    T_h=17,
    T_p=17,
    T_in_a_day=17,
    forecast_day_number=7,
    batch_size=64,
    val_rate=0.2,
    scale=True,
    shuffle_train=False,
):
    """
    返回：
        train_loader
        val_loader
        test_loader
        count_scale
    """

    print("train realtime OD forecast")
    train_dataset = RealTimeODIndexDataset(
        unfinished_flow_file=unfinished_flow_file,
        distribution_rate_file=distribution_rate_file,
        finished_od_file=finished_od_file,
        future_od_file=future_od_file,
        complete_od_file=future_od_file,

        T_h=T_h,
        T_p=T_p,
        T_in_a_day=T_in_a_day,
        forecast_day_number=forecast_day_number,

        is_train=True,
        is_val=False,
        val_rate=val_rate,

        scale=scale,
        count_scale=None,
    )

    count_scale = train_dataset.get_count_scale()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False,
    )

    print("val realtime OD forecast")
    val_dataset = RealTimeODIndexDataset(
        unfinished_flow_file=unfinished_flow_file,
        distribution_rate_file=distribution_rate_file,
        finished_od_file=finished_od_file,
        future_od_file=future_od_file,
        complete_od_file=future_od_file,

        T_h=T_h,
        T_p=T_p,
        T_in_a_day=T_in_a_day,
        forecast_day_number=forecast_day_number,

        is_train=True,
        is_val=True,
        val_rate=val_rate,

        scale=scale,
        count_scale=count_scale,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    print("test realtime OD forecast")
    test_dataset = RealTimeODIndexDataset(
        unfinished_flow_file=unfinished_flow_file,
        distribution_rate_file=distribution_rate_file,
        finished_od_file=finished_od_file,
        future_od_file=future_od_file,
        complete_od_file=future_od_file,

        T_h=T_h,
        T_p=T_p,
        T_in_a_day=T_in_a_day,
        forecast_day_number=forecast_day_number,

        is_train=False,
        is_val=False,
        val_rate=0,

        scale=scale,
        count_scale=count_scale,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, count_scale