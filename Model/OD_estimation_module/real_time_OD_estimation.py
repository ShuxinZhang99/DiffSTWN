import torch
from torch import nn
import torch.nn.functional as F


class LightGateFusion(nn.Module):
    """
    轻量门控融合，不使用 Conv1d。

    输入：
        long_unfinished_od:  [B, N*N, T_h]
        short_unfinished_od: [B, N*N, T_h]

    输出：
        fused_unfinished_od: [B, N*N, T_h]
    """

    def __init__(self):
        super(LightGateFusion, self).__init__()

        self.w_long = nn.Parameter(torch.tensor(0.0))
        self.w_short = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, long_unfinished_od, short_unfinished_od):
        gate = torch.sigmoid(
            self.w_long * long_unfinished_od
            + self.w_short * short_unfinished_od
            + self.bias
        )

        fused_unfinished_od = (
            gate * long_unfinished_od
            + (1.0 - gate) * short_unfinished_od
        )

        return fused_unfinished_od

class FusionConv(nn.Module):
    """
    分块版 FusionConv，避免 B*M 太大导致显存爆炸。

    输入：
        long_unfinished_od:  [B, N*N, T_h]
        short_unfinished_od: [B, N*N, T_h]

    输出：
        fused_unfinished_od: [B, N*N, T_h]
    """

    def __init__(
        self,
        device,
        hidden_channels=16,
        kernel_size=3,
        chunk_size=1024,
        
    ):
        super(FusionConv, self).__init__()

        self.chunk_size = chunk_size
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding
            )
        ).to(device)

    def forward(self, long_unfinished_od, short_unfinished_od):
        B, M, T = long_unfinished_od.shape

        x = torch.stack(
            [long_unfinished_od, short_unfinished_od],
            dim=2
        )
        # [B, M, 2, T]

        x = x.reshape(B * M, 2, T)
        # [B*M, 2, T]

        outs = []

        for start in range(0, B * M, self.chunk_size):
            end = min(start + self.chunk_size, B * M)

            x_chunk = x[start:end]
            out_chunk = self.net(x_chunk)

            outs.append(out_chunk)

        out = torch.cat(outs, dim=0)
        # [B*M, 1, T]

        fused_unfinished_od = out.reshape(B, M, T)

        return fused_unfinished_od


class RealTimeODEstimator(nn.Module):
    """
    实时 OD 需求矩阵估计模块。

    核心思想：
        1. 进站流序列为 [B, N, T]
        2. OD 分布率为 [B, N*N, T]
        3. 将 OD 分布率 reshape 为 [B, N, N, T]
        4. 通过广播操作计算：
              unfinished_od[o, d, t] = inflow[o, t] * rate[o, d, t]
        5. 再 flatten 为 [B, N*N, T]
    """

    def __init__(self, time_lag, num_stations, hidden_dim, kernel_size, device):
        super().__init__()

        self.time_lag     = time_lag
        self.num_stations = num_stations
        self.num_od_pairs = num_stations * num_stations

        self.fusion_conv = FusionConv(
            hidden_channels=hidden_dim,
            kernel_size=kernel_size,
            device=device
        )

        # self.fusion_conv = LightGateFusion()

    def expand_inflow_to_od(self, unfinished_inflow, distribution_rate):
        """
        将进站流序列和 OD 分布率相乘，得到未完成 OD 需求。

        参数：
            unfinished_inflow: [B, N, T]
            distribution_rate: [B, N*N, T]

        返回：
            unfinished_od: [B, N*N, T]
        """

        B, N, T = unfinished_inflow.shape
        M = N * N

        assert N == self.num_stations
        assert distribution_rate.shape == (B, M, T)

        # [B, N*N, T] -> [B, N, N, T]
        # 第二维是 origin，第三维是 destination
        rate = distribution_rate.reshape(B, N, N, T)

        # [B, N, T] -> [B, N, 1, T]
        # 在 destination 维度上广播
        inflow = unfinished_inflow.unsqueeze(2)

        # 广播乘法：
        # [B, N, 1, T] * [B, N, N, T] -> [B, N, N, T]
        unfinished_od = inflow * rate

        # [B, N, N, T] -> [B, N*N, T]
        unfinished_od = unfinished_od.reshape(B, M, T)

        return unfinished_od

    def forward(self, unfinished_inflow, long_term_rate, short_term_rate, finished_od):

        B, N, T = unfinished_inflow.shape
        M = N * N

        assert long_term_rate.shape == (B, M, T)
        assert short_term_rate.shape == (B, M, T)
        assert finished_od.shape == (B, M, T)

        # 长时未完成 OD 需求
        long_unfinished_od = self.expand_inflow_to_od(
            unfinished_inflow,
            long_term_rate
        )

        # 短时未完成 OD 需求
        short_unfinished_od = self.expand_inflow_to_od(
            unfinished_inflow,
            short_term_rate
        )

        # Fusion Conv
        fused_unfinished_od = self.fusion_conv(
            long_unfinished_od,
            short_unfinished_od
        )

        # 残差式补全：
        # 完整 OD 需求 = 已完成 OD 需求 + 估计的未完成 OD 需求
        complete_od = finished_od + fused_unfinished_od

        # OD 需求一般非负，可以根据需要启用
        # complete_od = F.relu(complete_od)

        return complete_od


