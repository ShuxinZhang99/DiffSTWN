import torch
from torch import nn
import torch.nn.functional as F
from Model.ST_Distribution_model import *
from Model.Attention import *
from Model.ST_Decompose import *
from Model.AGCN import *

class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=1024):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class ResidualBlock(nn.Module):
    def __init__(self, node_num, station_num, ext_num, o_graph, d_graph, con_time_length, con_pre_len, d_model, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv2d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )

        self.st_decompose = ST_Decomposition(moving_kernel_size=(1,3), d_model=d_model, num_ext_factors=ext_num,con_time_length=con_time_length,
                                             con_pre_len=con_pre_len, num_nodes=node_num, num_stations=station_num,
                                             origin_graph=o_graph, destination_graph=d_graph, num_layer=2)
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv2d(1, 2 * residual_channels, 1)
        self.output_projection = nn.Conv2d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.output_projection.weight)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)

    def forward(self, x, ext_inputs, conditioner, diffusion_step, o_adj, d_adj):
        # o_adj and d_adj: physical adjacency matrix
        B, _, V, T = x.shape

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1).unsqueeze(-1)  # (t, residual_channels, 1, 1)

        conditioner = self.st_decompose(conditioner, ext_inputs, o_adj, d_adj)  # (B, residual_channels, V, T)

        conditioner = self.conditioner_projection(conditioner).view(B, -1, V, T)  # (B, 2 * residual_channels, V, T)

        y = x + diffusion_step  # (B, Residual_channels, V, T)

        y = self.dilated_conv(y) + conditioner  # (B, 2 * residual_channels, V, T)

        gate, filter = torch.chunk(y, 2, dim=1)  # (B, residual_channels, V, T)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        node_num,
        station_num,
        ext_num,
        fore_horizon,
        target_len,
        cond_length,
        o_graph,
        d_graph,
        d_model,
        time_emb_dim=16,
        spa_emb_dim=16,
        residual_layers=1,
        residual_channels=2,
        dilation_cycle_length=2,
        residual_hidden=16,
        zinb_residual_channels=8,
        zinb_outputs_are_logits = False,
    ):
        super().__init__()

        self.device = device
        self.station_num = station_num
        self.fore_horizon = fore_horizon
        self.his_time_step = cond_length - fore_horizon

        # 自适应图生成
        self.adagraph = AdaptiveFullODGraph(
            num_origin=station_num,
            num_destination=station_num,
            embedding_dim=spa_emb_dim,
            top_k=20,
            temperature=1.0,
        ).to(device)

        self.input_projection = nn.Conv2d(1, residual_channels, 1).to(device)  # 处理进站流、稀疏OD流
        # self.input_projection = nn.Conv3d(1, residual_channels, 1).to(device)  # 处理高维OD数据
        
        self.diffusion_embedding = DiffusionEmbedding(time_emb_dim, proj_dim=residual_hidden).to(device)
        self.cond_upsampler = CondUpsampler(target_dim=target_len, cond_length=cond_length).to(device)

        # 残差块
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    node_num = node_num,
                    station_num=station_num,
                    ext_num=ext_num,
                    o_graph=o_graph,
                    d_graph=d_graph,
                    con_time_length=cond_length,
                    con_pre_len=target_len,
                    d_model=d_model,
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        ).to(device)

        self.skip_projection = nn.Conv2d(residual_channels, residual_channels, kernel_size=3, padding=1).to(device)
        self.output_projection = nn.Conv2d(residual_channels, 1, kernel_size=3, padding=1).to(device)
        self.linear = nn.Linear(cond_length, 256)
        self.linear2 = nn.Linear(256, target_len)
        self.zinb_model = NBNorm_ZeroInflated(c_in=1, c_out=1, his_time_step=self.his_time_step, fore_horizon=fore_horizon, hidden=zinb_residual_channels).to(device)
        self.zinb_output_are_logits = zinb_outputs_are_logits

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)
    
    def predict_zinb(self, history):
        """
        predict the zinb parameters。

        history:
            [B, F, V, T_h]

        returns:
            n_params, p, pi
        """
        n_params, p, pi = self.zinb_model(history)

        eps = 1e-6

        # 如果 zinb_model 内部已经进行了 softplus/sigmoid, 这里只需要做数值截断。
        n_params = n_params.clamp_min(eps)
        p = p.clamp(eps, 1.0 - eps)
        pi = pi.clamp(eps, 1.0 - eps)

        return n_params, p, pi

    def forward(self, inputs, ext_inputs, time, cond):
        # inputs and ext_inputs have the same dimension
        B, _, V, T = inputs.shape
        # self-adaptive graph construction
        o_adj, d_adj = self.adagraph()

        x = self.input_projection(inputs).reshape(B, -1, V, T)  # (B, Residual_channels, od_pairs, T)
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)  # (t, residual_hidden)
        cond_up = self.cond_upsampler(cond)  # (B, F, V, T)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, ext_inputs, cond_up, diffusion_step, o_adj, d_adj)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        # 预测
        x_output = self.output_projection(x)
        x_output = self.linear(x_output)
        x_output = F.relu(x_output)
        x_output = self.linear2(x_output)

        n_params, p, pi = self.predict_zinb(cond[:, :, :, :self.his_time_step])

        return x_output, n_params, p, pi