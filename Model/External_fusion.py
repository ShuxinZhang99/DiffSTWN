import torch
from torch import nn
import torch.nn.functional as F

# ----------------------------------------
# 1. 外部特征编码模块
# ----------------------------------------
class ExternalEncoder(nn.Module):
    def __init__(self, num_ext_factors, d_model):
        super().__init__()
        self.num_ext = num_ext_factors
        self.d_model = d_model

        # 公式 (30): 2D Convolution + Sigmoid 增强特征
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 公式 (32): 外部因子之间的 Multi-Head Self-Attention
        self.ext_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

        # 公式 (31): Concat 后的 MLP 映射层
        self.mlp = nn.Sequential(
            nn.Linear(num_ext_factors * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, ext_inputs):
        """
        :param ext_inputs: 外部特征输入, 形状期望为 (B, num_ext, T, d_model)
        :return: SE (Semantic External Features), 形状为 (B, 1, d_model)
        """
        B, N_ext, T, D = ext_inputs.shape
        z_list = []

        # 公式 (30): 对每个外部因子独立进行 Conv2D + Pooling
        for i in range(N_ext):
            e_i = ext_inputs[:, i:i + 1, :, :]  # (B, 1, T, D)
            z_i = self.conv2d(e_i)  # (B, 1, T, D)
            # 时间维度池化时间轴得到单一表征
            z_i = F.adaptive_avg_pool2d(z_i, (T, D)).squeeze(2).squeeze(1)  # (B, T, D)
            z_list.append(z_i)

        z = torch.stack(z_list, dim=1)  # (B, num_ext, T, D)
        z = z.reshape(B, T, -1)

        # MultiHeadAttn (Q = z_i, K = z, V = z)
        A, _ = self.ext_attn(query=z, key=z, value=z)  # (B, T, num_ext * D)
        # 通过 MLP
        SE = self.mlp(A).unsqueeze(1)  # (B, T, D)

        return SE


# ---------------------------------
# 2. ST 与 SE 交叉注意力融合模块
# ---------------------------------
class ST_SE_CrossFusion(nn.Module):
    def __init__(self, num_nodes, d_model):
        super().__init__()
        # 公式 (34): As - ST 自注意力
        self.attn_s = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        # 公式 (35): Ac - ST 与 SE 交叉注意力
        self.attn_c = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

        # 公式 (33): Concat[As, Ac] 后的投影 MLP
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_nodes * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_nodes)
        )

    def forward(self, ST, SE):
        """
        :param ST: 时空特征, 形状为 (B, T, V, d_model)
        :param SE: 外部语义特征, 形状为 (B, T, V, d_model)
        :return: CR (Conditional Representation), 形状为 (B, T, V)
        """
        B, T, V, D = ST.shape
        ST = ST.reshape(B, T, -1)
        SE = ST.reshape(B, T, -1)
        # As = MultiHeadAttn(Q=ST, K=ST, V=ST)
        A_s, _ = self.attn_s(query=ST, key=ST, value=ST)  # (B, T, V*d_model)

        # Ac = MultiHeadAttn(Q=ST, K=SE, V=SE)
        A_c, _ = self.attn_c(query=ST, key=SE, value=SE)  # (B, T, V*d_model)

        # 公式 (33): CR = MLP(Concat[As, Ac])
        concat_A = torch.cat([A_s, A_c], dim=-1)  # (B, T, 2 * V * d_model)
        CR = self.mlp(concat_A)  # (B, T, V)

        return CR