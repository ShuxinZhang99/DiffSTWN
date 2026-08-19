import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_array(file_path):
    """
    支持读取 .npy / .csv  文件。
    """
    file_path = Path(file_path).expanduser()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        data = np.load(file_path)
    elif suffix == ".csv":
        data = pd.read_csv(file_path, index_col=0)
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return data


def flatten_od_if_needed(data):
    """
        将 OD 数据统一转成 [N*N, T]。

        支持：
            [N*N, T]
            [N, N, T]
            [T, N, N]
    """
    data = np.asarray(data)

    if data.ndim == 2:
        return data

    if data.ndim == 3:
        # [N, N, T]
        if data.shape[0] == data.shape[1]:
            N = data.shape[0]
            T = data.shape[2]
            return data.reshape(N * N, T)

        # [T, N, N]
        if data.shape[1] == data.shape[2]:
            T = data.shape[0]
            N = data.shape[1]
            data = np.transpose(data, (1, 2, 0))
            return data.reshape(N * N, T)

    raise ValueError(
        f"OD data shape should be [N*N, T], [N, N, T] or [T, N, N], but got {data.shape}"
    )


class RealTimeODIndexDataset(Dataset):
    """
    实时 OD 需求估计 + 未来 OD 需求预测数据集。

    原始数据形状：
        unfinished_flow:   [N, T_total]
        distribution_rate: [N*N, T_total]
        finished_od:       [N*N, T_total]
        future_od:         [N*N, T_total]

    给定当前 index：
        unfinished_flow:
            [:, index - T_h : index]
        long_term_rate:
            [:, index - 7*T_in_a_day - T_h : index - 7*T_in_a_day]
        short_term_rate:
            [:, index - T_in_a_day - T_h : index - T_in_a_day]
        finished_od:
            [:, index - T_h : index]
        label / future_od:
            [:, index : index + T_p]

    单个样本输出：
        unfinished_flow: [N, T_h]
        long_term_rate:  [N*N, T_h]
        short_term_rate: [N*N, T_h]
        finished_od:     [N*N, T_h]
        future_od:       [N*N, T_p]
    """
    def __init__(
            self,
            unfinished_flow_file,
            distribution_rate_file,
            finished_od_file,
            future_od_file,
            complete_od_file=None,

            T_h=17,
            T_p=17,
            T_in_a_day=17,
            forecast_day_number=7,

            is_train=True,
            is_val=False,
            val_rate=0.2,

            scale=True,
            count_scale=None
    ):
        super().__init__()

        self.T_h = T_h
        self.T_p = T_p
        self.T_in_a_day = T_in_a_day
        self.forecast_day_number = forecast_day_number
        self.test_len = self.T_in_a_day * self.forecast_day_number

        self.is_train = is_train
        self.is_val = is_val
        self.val_rate = val_rate
        self.scale = scale

        # =============
        # 1. 读取数据
        # =============
        self.unfinished_flow = load_array(unfinished_flow_file)
        # self.unfinished_flow = self.unfinished_flow[0, :198, :]
        self.distribution_rate = flatten_od_if_needed(load_array(distribution_rate_file))
        self.finished_od = flatten_od_if_needed(load_array(finished_od_file))
        self.future_od = flatten_od_if_needed(load_array(future_od_file))

        self.complete_od = None
        if complete_od_file is not None:
            self.complete_od = flatten_od_if_needed(load_array(complete_od_file))

        # =============
        # 2. 检查维度
        # =============
        if self.unfinished_flow.ndim != 2:
            raise ValueError(
                f"unfinished_flow should be [N, T], but got {self.unfinished_flow.shape}"
            )
        self.N, self.total_time = self.unfinished_flow.shape
        self.num_od_pairs = self.N * self.N

        if self.distribution_rate.shape != (self.num_od_pairs, self.total_time):
            raise ValueError(
                f"distribution_rate should be [{self.num_od_pairs}, {self.total_time}], "
                f"but got {self.distribution_rate.shape}"
            )

        if self.finished_od.shape != (self.num_od_pairs, self.total_time):
            raise ValueError(
                f"finished_od should be [{self.num_od_pairs}, {self.total_time}], "
                f"but got {self.finished_od.shape}"
            )

        if self.future_od.shape != (self.num_od_pairs, self.total_time):
            raise ValueError(
                f"future_od should be [{self.num_od_pairs}, {self.total_time}], "
                f"but got {self.future_od.shape}"
            )

        if self.complete_od is not None:
            if self.complete_od.shape != (self.num_od_pairs, self.total_time):
                raise ValueError(
                    f"complete_od should be [{self.num_od_pairs}, {self.total_time}], "
                    f"but got {self.complete_od.shape}"
                )

        # ============
        # 3.构造index
        # ============
        # index 至少要保证 long_term_rate 可以切到：
        # index - 7*T_in_a_day - T_h >= 0
        self.start_index = 7 * self.T_in_a_day + self.T_h

        # index 最大可以等于 total_time，因为切片是 [:, index - T_h : index]
        self.end_index = self.total_time - self.T_p

        if self.start_index >= self.end_index:
            raise ValueError(
                f"Invalid index range: start_index={self.start_index}, "
                f"end_index={self.end_index}. "
                f"Please check T_h={self.T_h}, T_p={self.T_p}, T_in_a_day={self.T_in_a_day}, "
                f"total_time={self.total_time}."
            )

        all_indices = list(range(self.start_index, self.end_index + 1))

        # 最后 forecast_day_number 天作为测试集
        test_start_index = self.total_time - self.test_len

        if test_start_index <= self.start_index:
            raise ValueError(
                f"test_start_index={test_start_index} <= start_index={self.start_index}. "
                f"Please reduce forecast_day_number or T_in_a_day."
            )

        # 为避免训练标签进入测试区间，要求：
        # train/val 的 index + T_p <= test_start_index
        train_val_indices = [
            idx for idx in all_indices
            if idx + self.T_p <= test_start_index
        ]

        # 测试集 index 从 test_start_index 开始，且已经保证 index + T_p <= total_time
        test_indices = [
            idx for idx in all_indices
            if idx >= test_start_index
        ]

        if self.is_train:
            if len(train_val_indices) == 0:
                raise ValueError(
                    "No train/val samples. Please reduce T_h, T_p, "
                    "forecast_day_number or T_in_a_day."
                )

            val_len = int(len(train_val_indices) * self.val_rate)
            train_len = len(train_val_indices) - val_len

            if self.is_val:
                self.indices = train_val_indices[train_len:]
            else:
                self.indices = train_val_indices[:train_len]
        else:
            if len(test_indices) == 0:
                raise ValueError(
                    "No test samples. Please reduce T_p or forecast_day_number."
                )

            self.indices = test_indices

        # ==========
        # 4.归一化
        # ==========
        # distribution_rate 是比例，不归一化。
        # unfinished_flow / finished_od / future_od 是需求量，统一除以 count_scale。
        if self.scale:
            if count_scale is None:
                # 只用训练测试分界点之前的数据计算 scale，避免明显的数据泄漏
                scale_end = test_start_index

                max_list = [
                    np.max(self.unfinished_flow[:, :scale_end]),
                    np.max(self.finished_od[:, :scale_end]),
                    np.max(self.future_od[:, :scale_end]),
                ]

                if self.complete_od is not None:
                    max_list.append(np.max(self.complete_od[:, :scale_end]))

                self.count_scale = max(max_list)

                if self.count_scale == 0:
                    self.count_scale = 1.0
            else:
                self.count_scale = count_scale

            self.unfinished_flow = self.unfinished_flow / self.count_scale
            self.finished_od = self.finished_od / self.count_scale
            self.future_od = self.future_od / self.count_scale
            if self.complete_od is not None:
                self.complete_od = self.complete_od / self.count_scale
        else:
            self.count_scale = 1.0

        print(
            "RealTimeODIndexDataset | "
            f"is_train={self.is_train}, is_val={self.is_val}, "
            f"samples={len(self.indices)}, "
            f"N={self.N}, OD_pairs={self.num_od_pairs}, "
            f"T_h={self.T_h}, T_p={self.T_p}, total_time={self.total_time}"
        )

    def get_count_scale(self):
        return self.count_scale

    def __getitem__(self, item):
        index = self.indices[item]

        # 当前未完成进站流 [N, T_h]
        unfinished_flow = self.unfinished_flow[:, index - self.T_h: index]

        # 长时分布率 [N*N, T_h]
        long_term_rate = self.distribution_rate[:, index - 7 * self.T_in_a_day - self.T_h : index - 7 * self.T_in_a_day]

        # 短时分布率 [N*N, T_h]
        short_term_rate = self.distribution_rate[:, index - self.T_in_a_day - self.T_h : index - self.T_in_a_day]

        # 已完成OD需求 [N*N, T_h]
        finished_od = self.finished_od[:, index - self.T_h: index]

        future_od = self.future_od[:, index: index + self.T_p]

        sample = {
            "index": torch.tensor(index, dtype=torch.long),

            "unfinished_flow": torch.tensor(unfinished_flow, dtype=torch.float32),
            "long_term_rate": torch.tensor(long_term_rate, dtype=torch.float32),
            "short_term_rate": torch.tensor(short_term_rate, dtype=torch.float32),
            "finished_od": torch.tensor(finished_od, dtype=torch.float32),
            # 预测标签
            "future_od": torch.tensor(future_od, dtype=torch.float32),
        }

        # 如果有真实 complete_od，就作为监督标签
        if self.complete_od is not None:
            complete_od = self.complete_od[:, index - self.T_h: index]

            sample["complete_od"] = torch.tensor(complete_od, dtype=torch.float32)

        return sample

    def __len__(self):
        return len(self.indices)
    




