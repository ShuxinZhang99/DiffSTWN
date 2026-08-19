import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Attention import *
from Model.GCN_layer import *
from Model.AGCN import *
from Model.DGCN import *
from Model.External_fusion import *


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        # self.device = device

    def forward(self, inputs):
        '''
        :param inputs: [batch_size, seq_len, d_model]
        :return:
        '''
        residual = inputs
        output = self.fc(inputs)
        return (output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = AttentionLayer(ProbAttention(), d_model=d_model, n_heads=n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, flow_time):
        '''
        :param flow_week:
        :param flow_day:
        :param flow_time: [batch_size, num_station, d_model]
        :return:
        '''
        enc_outputs, attn = self.enc_self_attn(flow_time, flow_time, flow_time)  # (batch, time_steps, d_model)
        enc_outputs = self.pos_ffn(enc_outputs).permute(0, 2, 1)  # (batch, d_model, time_step)
        return enc_outputs, attn


class moving_avg(nn.Module):
    '''
    Moving average block to highlight the trend of time series
    '''
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # 对时间序列两端进行零填充，以便计算移动平均，通过在时间序列的开头和结尾分别复制第一个和最后一个时间步得到
        # x.shape: (batch_size, seq_len, features)
        x = x.permute(0, 3, 1, 2)  # (B, T, F, V)
        front = x[:, 0:1, :, :].repeat(1, (self.kernel_size[1] - 1) // 2, 1, 1)
        end = x[:, -1:, :, :].repeat(1, (self.kernel_size[1] - 1) // 2, 1, 1)
        # 将零填充后的序列与原始输入序列连接起来，以便计算移动平均时考虑序列两端的信息
        x = torch.cat([front, x, end], dim=1)
        # print(x.shape)
        x = self.avg(x.permute(0, 2, 3, 1))  # (B, F, V, T)
        # x = x.permute(0, 3, 1, 2)  # (B, T, F, V)

        return x


class series_decomp(nn.Module):
    '''
    series decomposition block
    '''

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        # print(x.shape, moving_mean.shape)
        res = x - moving_mean
        return res, moving_mean


class ST_Decomposition(nn.Module):
    def __init__(self, moving_kernel_size, d_model, num_ext_factors, con_time_length, con_pre_len, num_nodes, num_stations, origin_graph, destination_graph, num_layer):  # d_adj_mx, max_diffusion_step,
        """
        :params: moving_kernel_size: moving_avg
        :params: num_inputs: time_step
        :params: num_units: the output size of the dgcn
        :params: num_channels: channel lists: [channel_1, ..., channel_n]
        :params: o_adj_mx: origin_based matrix
        :params: d_adj_mx: destination_based matrix
        :params: max_diffusion_step: diffusion step
        :params: num_nodes: number of od nodes
        """
        super().__init__()
        self.mov_kernel_size = moving_kernel_size
        self.time_step = con_time_length
        self.pre_len = con_pre_len  # output channels 等于输出的时间步
        # self.max_diffusion_step = max_diffusion_step
        self.num_nodes = num_nodes
        self.num_stations = num_stations
        self.num_layer = num_layer
        self.d_model = d_model
        # raw_graph
        self.origin_graph = origin_graph
        self.destination_graph = destination_graph

        self.series_decompose = series_decomp(kernel_size=self.mov_kernel_size)
        # ST_modeling
        self.embedding = nn.Conv1d(num_nodes, self.d_model, kernel_size=1)
        self.attention = EncoderLayer(d_model=self.d_model, d_ff=256, n_heads=8)
        self.projection = nn.Conv1d(self.d_model, num_nodes, kernel_size=1)
        # self.gcn = GraphConvolution(in_features=self.time_step, out_features=self.pre_len, num_nodes=num_nodes)
        self.adgcn =AdaptiveFullODGraphConv(in_channels=1, out_channels=1, num_origin=num_stations, num_destination=num_stations)
        # self.dgcn_block = DGCN(num_units=self.time_step, o_adj_mx=self.o_adj, d_adj_mx=self.d_adj,
        #                        max_diffusion_step=self.max_diffusion_step,
        #                        num_nodes=self.num_nodes)
        self.fusion = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.fusion1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.linear1 = nn.Linear(in_features=self.time_step, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=self.pre_len)

        # external_modeling
        self.ext_embedding = nn.Embedding(num_embeddings=32, embedding_dim=d_model)
        self.ext_encoder = ExternalEncoder(num_ext_factors=num_ext_factors, d_model=self.d_model)
        # 2. 将 x_st 维度线性映射至 d_model，用于 Attention 计算
        self.st_to_dmodel = nn.Linear(num_nodes, d_model)
        # 3. 时空特征与外部特征交叉融合模块
        self.st_se_fusion = ST_SE_CrossFusion(num_nodes=num_nodes, d_model=self.d_model)

    def forward(self, inputs, ext_inputs, o_adj=None, d_adj=None):
        # inputs.shape: (B, F, V, T)
        # ext_inputs: (B, F, N_ext, T_p)
        x_se, x_tr = self.series_decompose(inputs)

        for i in range(self.num_layer):
            # ST_modeling
            res = x_se  # (B, F, V, T)
            # seasonal temporal part
            x_se = self.embedding(x_se.reshape(x_se.size()[0], -1, x_se.size()[3]))
            x_se, _ = self.attention(x_se)
            x_se = self.projection(x_se).unsqueeze(1)  # (B, F, V, T)

            # seasonal spatial part
            x_spa = self.adgcn(res, self.origin_graph, self.destination_graph)
            # adaptive graph convolutional
            if o_adj is not None and d_adj is not None:
                x_spa = self.adgcn(x_spa, o_adj, d_adj)
            # x_spa = x_spa.unsqueeze(1)  # B, F, V, T

            x = torch.cat([x_se, x_spa], dim=1)  # B, 2*F, V ,T
            x = self.fusion(x)  # B, F, V, T
            x = self.fusion1(x)
            
            x_se, x_tr_sub = self.series_decompose(x)
            x_tr += x_tr_sub

        x_se_sum = self.linear1(x_se)
        x_se_sum = F.leaky_relu(x_se_sum, 0.4)
        x_se_sum = self.linear2(x_se_sum)
        x_st = x_tr + x_se_sum  # (B, F, V, T)

        # external_modeling
        ST_raw = x_st.permute(0, 3, 2, 1)  # (B, T, V, F)
        _, _, V, _ = ST_raw.shape
        ext_fea = self.ext_embedding(ext_inputs.squeeze(1))  # (B, N_ext, T, d_model)
        SE = self.ext_encoder(ext_fea).squeeze(1)        # (B, T, d_model)
        ST_proj = self.st_to_dmodel(ST_raw.squeeze(-1))  # (B, T, d_model)

        # Cross-attention
        CR = self.st_se_fusion(ST_proj, SE).permute(0, 2, 1).unsqueeze(1)  # (B, 1, V, T)

        return CR


