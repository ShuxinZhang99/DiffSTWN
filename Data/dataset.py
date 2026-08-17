# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import Dataset

def search_recent_data(train, label_start_idx, T_p, T_h):
    """
    T_p: prediction time steps
    T_h: historical time steps
    """
    if label_start_idx + T_p > len(train): return None
    start_idx, end_idx = label_start_idx - T_h, label_start_idx - T_p + T_p
    if start_idx < 0 or end_idx < 0: return None
    return (start_idx, end_idx), (label_start_idx, label_start_idx + T_p)


class CleanDataset():
    def __init__(self, feature_file, real_file, val_start_idx, external_file=None, normalization_type="minmax", eps=1e-8):

        self.feature_file = feature_file
        self.real_file    = real_file
        self.external_file = external_file
        self.val_start_idx = val_start_idx
        self.normalization_type = normalization_type.lower()
        self.eps = eps

        # 初始化归一化参数
        self.data_min = None
        self.data_max = None
        self.data_scale = None
        self.mean = None
        self.std = None

        # raw_label：原始计数标签
        # label：归一化标签
        # feature：归一化历史输入
        (
            self.raw_label,
            self.label,
            self.feature,
            self.external_feature,
        ) = self.read_data()

    def read_data(self):
        """
        Returns
        -------
        raw_label:
            原始非负计数数据，[T,V,F]。

        normalized_label:
            归一化后的标签，[T,V,F]。

        normalized_feature:
            归一化后的输入特征，[T,V,F]。
        """

        data      = np.load(self.feature_file)[:, :]
        real_data = np.load(self.real_file)[:, :]

        # 统一为 [T,V,F]
        if data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        if real_data.ndim == 2:
            real_data = np.expand_dims(real_data, axis=-1)

        if data.ndim != 3:
            raise ValueError(
                "输入数据应该为 [T,V] 或 [T,V,F],"
                f"当前形状为 {data.shape}"
            )

        # 清理 NaN 和 Inf
        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        real_data = np.nan_to_num(
            real_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ZINB 只适用于非负计数
        if np.any(data < 0):
            data = np.clip(
                data,
                a_min=0.0,
                a_max=None,
            )
        if np.any(real_data < 0):
            real_data = np.clip(
                real_data,
                a_min=0.0,
                a_max=None,
            )

        data, real_data = data.astype(np.float32), real_data.astype(np.float32)


        # 保留原始计数标签
        raw_label = real_data.copy()

        # 只计算一次归一化
        normalized_data = self.normalization(
            data
        ).astype(np.float32)
        normalized_real_data = self.normalization(
            real_data
        ).astype(np.float32)

        # label 和 feature 分开复制，避免后续原地修改相互影响
        normalized_label = normalized_real_data.copy()
        normalized_feature = normalized_data.copy()

        if self.external_file is not None:
            ext_data = np.load(self.external_file)[:, :]

            # 与 feature_file 一样的维度整理
            if ext_data.ndim == 2:
                ext_data = np.expand_dims(ext_data, axis=-1)

            # 与 feature_file 一样的 NaN 和 Inf 清理
            ext_data = np.nan_to_num(
                ext_data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # 转为 float32（不进行 normalization）
            external_feature = ext_data.astype(np.float32)
        else:
            external_feature = None

        return (
            raw_label,
            normalized_label,
            normalized_feature,
            external_feature,
        )


    def normalization(self, feature):
        """
        使用训练集统计量对完整数据归一化。

        支持：
            minmax
            max
            zscore
            log1p
            none
        """

        train = feature[:self.val_start_idx]

        if train.size == 0:
            raise ValueError(
                "训练集为空，请检查 val_start_idx。"
            )

        if self.normalization_type == "minmax":
            # 所有统计量只从训练集获得
            self.data_min = float(np.min(train))
            self.data_max = float(np.max(train))

            self.data_scale = (
                self.data_max - self.data_min
            )

            if self.data_scale < self.eps:
                warnings.warn(
                    "训练集最大值和最小值几乎相同，"
                    "归一化尺度已设置为 1。"
                )
                self.data_scale = 1.0

            normalized = (
                feature - self.data_min
            ) / self.data_scale

            return normalized

        elif self.normalization_type == "max":
            # 非负计数数据也可以直接除以训练集最大值
            self.data_min = 0.0
            self.data_max = float(np.max(train))

            self.data_scale = self.data_max

            if self.data_scale < self.eps:
                self.data_scale = 1.0

            return feature / self.data_scale

        elif self.normalization_type == "zscore":
            self.mean = float(np.mean(train))
            self.std = float(np.std(train))

            if self.std < self.eps:
                self.std = 1.0

            return (
                feature - self.mean
            ) / self.std

        elif self.normalization_type == "log1p":
            # log1p 不需要训练集统计量
            return np.log1p(feature)

        elif self.normalization_type == "none":
            return feature.copy()

        else:
            raise ValueError(
                "不支持的 normalization_type："
                f"{self.normalization_type}。"
                "可选值为 minmax、max、zscore、"
                "log1p、none。"
            )

    def reverse_normalization(self, x):
        """
        将扩散模型的归一化预测恢复到原始客流尺度。

        同时兼容：
            NumPy array
            PyTorch tensor

        注意：
        ZINB 生成的 n、p、pi 不能调用这个函数。
        根据 n、p、pi 计算出的均值本身就在原始计数尺度。
        """

        if self.normalization_type == "minmax":
            return (
                x * self.data_scale
                + self.data_min
            )

        elif self.normalization_type == "max":
            return x * self.data_scale

        elif self.normalization_type == "zscore":
            return (
                x * self.std
                + self.mean
            )

        elif self.normalization_type == "log1p":
            if torch.is_tensor(x):
                return torch.expm1(x)

            return np.expm1(x)

        elif self.normalization_type == "none":
            return x

        else:
            raise ValueError(
                "不支持的 normalization_type："
                f"{self.normalization_type}"
            )


class TrafficDataset(Dataset):
    """
    每个样本返回：

        label:
            归一化未来标签，用于 diffusion loss。

        node_feature:
            归一化历史输入。

        pos_w:
            星期位置。

        pos_d:
            日内位置。

        raw_label:
            原始未来计数，用于 ZINB NLL。
    """
    def __init__(self, clean_data, data_range, T_h, T_p, V, points_per_hour):
        self.T_h = T_h
        self.T_p = T_p
        self.V = V
        self.points_per_hour = points_per_hour
        self.data_range = data_range

        # 归一化未来标签
        self.label = np.asarray(
            clean_data.label,
            dtype=np.float32,
        )

        # 原始计数未来标签
        self.raw_label = np.asarray(
            clean_data.raw_label,
            dtype=np.float32,
        )

        # 归一化历史输入
        self.feature = np.asarray(
            clean_data.feature,
            dtype=np.float32,
        )

        if clean_data.external_feature is not None:
            self.external_feature = np.asarray(
                clean_data.external_feature,
                dtype=np.float32,
            )
        else:
            self.external_feature = None

        if self.label.shape != self.raw_label.shape:
            raise ValueError(
                "归一化标签和原始标签形状不一致："
                f"{self.label.shape} vs "
                f"{self.raw_label.shape}"
            )

        if self.feature.shape != self.label.shape:
            raise ValueError(
                "输入特征和标签形状不一致："
                f"{self.feature.shape} vs "
                f"{self.label.shape}"
            )

        if self.feature.shape[1] != self.V:
            raise ValueError(
                f"配置节点数 V={self.V}，"
                f"但数据节点数为 {self.feature.shape[1]}"
            )

        # 准备滑动窗口索引
        self.idx_lst = self.get_idx_lst()

        print(
            "sample num:",
            len(self.idx_lst),
        )

    def __getitem__(self, index):
        recent_idx = self.idx_lst[index]

        # recent_idx[1]：未来预测区间
        label_start, label_end = (
            recent_idx[1][0],
            recent_idx[1][1],
        )

        # 归一化未来标签
        label = self.label[
            label_start:label_end
        ]

        # 原始未来计数标签
        raw_label = self.raw_label[
            label_start:label_end
        ]

        # recent_idx[0]：历史输入区间
        history_start, history_end = (
            recent_idx[0][0],
            recent_idx[0][1],
        )

        node_feature = self.feature[
            history_start:history_end
        ]

        pos_w, pos_d = self.get_time_pos(
            history_start
        )

        pos_w = np.asarray(
            pos_w,
            dtype=np.int32,
        )

        pos_d = np.asarray(
            pos_d,
            dtype=np.int32,
        )

        if self.external_feature is not None:
            ext_feature = self.external_feature[
                          history_start:history_end
                          ]
            # 未来预测时间段的外部特征 [T_p, V, F_ext]
            ext_label = self.external_feature[
                        label_start:label_end
                        ]
            return (
                label,
                node_feature,
                pos_w,
                pos_d,
                raw_label,
                ext_feature,
                ext_label,
            )

        return (
            label,
            node_feature,
            pos_w,
            pos_d,
            raw_label,
        )


    def __len__(self):
        return len(self.idx_lst)

    def get_time_pos(self, idx):
        idx = np.array(range(self.T_h)) + idx
        pos_w = (idx // (self.points_per_hour * 24)) % 7  # day of week
        pos_d = idx % (self.points_per_hour * 24)  # time of day
        return pos_w, pos_d

    def get_idx_lst(self):
        idx_lst = []
        start = self.data_range[0]
        end = self.data_range[1] if self.data_range[1] != -1 else self.feature.shape[0]

        for label_start_idx in range(start, end):
            if label_start_idx % (24 * 6) < (7 * 6):
                continue
            if label_start_idx % (24 * 6) > (24 * 6) - self.T_p:
                continue

            recent = search_recent_data(self.feature, label_start_idx, self.T_p, self.T_h)  # recent data

            if recent:
                idx_lst.append(recent)
        return idx_lst
