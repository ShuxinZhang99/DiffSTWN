import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Masking import TriangularCausalMask, SpatialMask
from math import sqrt


class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)

    def forward(self, queries, keys, values, adj_mx=None, attn_mask=None):
        B, L, H, E = queries.shape  # (batch_size, seq_len, n_head, d_k )
        _, S, _, D = values.shape  # (batch_size, seq_len, n_heads, d_v)
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum('blhe, bshe -> bhls', queries, keys)  # （batch_size, n_heads, seq_len , d_k）
        if self.mask_flag:
            if attn_mask is None:
                # attn_mask = TriangularCausalMask(B, L)
                attn_mask = SpatialMask(B, L)

            scores.masked_fill_(attn_mask.mask, -np.inf)  # mask中取值为True位置对应于self的相应位置用负无穷表示

        A = torch.softmax(scale * scores, dim=-1)
        # print(A.shape)
        A = self.dropout(A)
        V = torch.einsum('bhls, bshd -> blhd', A, values)  # (batch_size, seq_len * station, head, d_v)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# 首先对K进行采样，得到K_sample,然后对q关于K_sample求M值，选取M值最大的u个q，对Top_u关于K求score值
class ProbAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sample Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real_U = U_part(factor*ln(L_K))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):  # 初始分值
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        return context

    def _update_context(self, context_in, V, scores, index):  # 更新分值
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index)

        return context.contiguous(), attn  # .transpose(2, 1)


# 与多头数量更改相关
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,  d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        # self.padding = 2
        # self.causul_convolution = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 3), stride=1, padding=(0, self.padding))
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # time_lag --> d_keys * n_heads
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, T = queries.shape  # (batch_size, d_model, time_steps)
        _, _, S = keys.shape
        H = self.n_heads
        # print(B, L, N, S)

        queries = self.query_projection(queries.permute(0, 2, 1)).view(B, T, H, -1)  # (batch_size, seq_len, n_head, d_k)
        keys = self.key_projection(keys.permute(0, 2, 1)).view(B, T, H, -1)
        values = self.value_projection(values.permute(0, 2, 1)).view(B, S, H, -1)

        # print(queries.shape, keys.shape, values.shape)

        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )
        out = out.view(B, T, -1)  # (batch_size, seq_len, d_values * n_heads)

        return self.out_projection(out), attn
