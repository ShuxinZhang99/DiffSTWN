import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from Model.ZeroInflated_NB_Loss import *
from Model.STWN import *

def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)  # 在最后一个维度（-1）上收集索引为t的元素
    return c.reshape(-1, 1, 1, 1)


class DiffWave(nn.Module):
    """
    Masked Diffusion Model
    """

    def __init__(self, N, sample_steps, sample_strategy, origin_graph, destination_graph, d_model, beta_schedule, fore_horizon, con_time_length, con_pre_len,
                 num_nodes, num_stations, num_ext, device):
        super(DiffWave, self).__init__()

        self.N = N                               # steps in the forward process
        self.sample_steps = sample_steps         # steps in the backward process
        self.sample_strategy = sample_strategy   # sample_strategy
        self.beta_start = 0.0002
        self.beta_end = 0.02
        self.beta_schedule = beta_schedule
        self.device = device
        self.fore_horizon = fore_horizon
        self.his_time_step = con_time_length - fore_horizon

        # =====================
        # 1. Denoising Network
        # =====================
        self.eps_model_3 = EpsilonTheta(node_num=num_nodes, station_num=num_stations, ext_num=num_ext, fore_horizon=fore_horizon, 
                                        target_len=con_pre_len, cond_length=con_time_length, o_graph=origin_graph, d_graph=destination_graph,
                                        d_model=d_model)

        # ===============================
        # 2. diffusion variance schedule
        # ===============================
        if self.beta_schedule == 'uniform':
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.N).to(self.device)

        elif self.beta_schedule == 'quad':
            self.beta = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.N) ** 2
            self.beta = self.beta.to(self.device)

        else:
            raise NotImplementedError

        self.alpha = (1.0 - self.beta).to(self.device)

        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # 将该输入张量中的元素进行累积乘积，依次将每个元素与前一个元素相乘

        self.sigma2 = self.beta

    def to_diffusion_format(self, x):
        """
        将 [B, N*N, T]转成[B, 1, N*N, T]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(
                f"Diffusion input should be [B, N*N, T] or [B, F, N*N, T], "
                f"but got {x.shape}"
            )
        return x

    def q_xt_x0(self, x_0, t, eps: Optional[torch.Tensor] = None):
        '''
        Sample from q(x_t|x_0) ~ N(x_t; \sqrt\bar\alpha_t * x_0, (1-\bar\alpha_t)I)
        前向处理，给数据加入服从高斯分布的 噪声
        '''
        if eps is None:
            eps = torch.randn_like(x_0)
        x_0 = x_0.to(self.device)
        t = t.to(self.device)

        mean = gather(self.alpha_bar, t) ** 0.5 * x_0
        var = 1 - gather(self.alpha_bar, t)
        # print('mean:', mean.shape, 'var:', var.shape)

        return mean + eps * (var ** 0.5)  # 生成服从均值为mean，方差为var的随机变量

    def p_sample(self, xt, ext_t, t, c):
        '''
        sample from p(x_(t-1)|x_t, c)，降噪还原，采样
        由x_t 恢复到 x_0
        '''
        xt = xt.to(self.device)
        t = t.to(self.device)
        c = c.to(self.device)  # c represents the x_masked
        eps_theta, _, _, _ = self.eps_model_3(xt, ext_t, t, c)
        alpha_coef = 1. / (gather(self.alpha, t) ** 0.5)
        eps_coef = gather(self.beta, t) / ((1 - gather(self.alpha_bar, t)) ** 0.5)
        mean = alpha_coef * (xt - eps_coef * eps_theta)

        # var = gather(self.sigma2, t)
        var = (1 - gather(self.alpha_bar, t - 1)) / (1 - gather(self.alpha_bar, t)) * gather(self.beta, t)

        if t[0] > 1:
            eps = torch.randn(xt.shape, device=xt.device)
        else:
            eps = torch.zeros(xt.shape, device=xt.device)

        return mean + eps * (var ** 0.5)

    def p_sample_loop(self, c, ext_inputs):  # 常规采样
        '''
        :param c: is the masked input tensor, (B, T, V, D), in the prediction task, T = T_h + T_p
        :param ext_inputs: external_factors_data, (B, T, V, D), in the prediction task, T = T_h + T_p
        :return x: 由x_t一直迭代到x_0, the predicted output tensor, (B, T, V, D)
        '''
        B, F, V, T = c.shape
        with torch.no_grad():
            x = torch.randn([B, F, V, T])  # generate input Gaussian white noise
            #  Remove noise for $T$ steps
            for t in range(self.N, 0, -1):  # t should start from T, and end at 1
                t = t - 1  # in code, t is index, so t should minus 1
                if t > 0:  # 不断更新迭代x，直到x0
                    x = self.p_sample(x, ext_inputs, x.new_full((B,), t, dtype=torch.int64), c)
        return x

    def p_sample_loop_ddim(self, c, ext_inputs):  # 加速采样策略
        x_masked = c.to(self.device)
        B, F, V, T = x_masked.shape

        N = self.N  # 前向加噪音步骤
        backward_steps = self.sample_steps  # 反向降噪步骤
        skip_type = self.beta_schedule
        if skip_type == 'uniform':
            skip = N // backward_steps
            seq = range(0, N, skip)
        elif skip_type == 'quad':
            seq = (np.linspace(0, np.sqrt(N * 0.8), backward_steps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        x = torch.randn([B, F, V, T], device=self.device)  # generate input noise，默认为高斯分布噪声
        xs, x0_preds = generalized_steps(x, ext_inputs, seq, self.eps_model_3, self.beta, c)
        return xs, x0_preds

    def set_sample_strategy(self, sample_strategy):
        self.sample_strategy = sample_strategy

    def set_ddim_sample_steps(self, sample_steps):
        self.sample_steps = sample_steps

    def evaluate(self, input, ext_inputs, n_samples):
        x_masked = input.to(self.device)
        B, F, V, T = x_masked.shape

        if n_samples < 1:
            raise ValueError(
                f"n_samples must larger than 1, now is {n_samples}"
            )
        
        # calculate the parameters of ZINB
        n_params, p, pi = self.eps_model_3.zinb_model(x_masked[:, :, :, :self.his_time_step])

        n_params = n_params.clamp_min(1e-6)
        p = p.clamp(1e-6, 1.0 - 1e-6)
        pi = pi.clamp(1e-6, 1.0 - 1e-6)

        if self.sample_strategy == 'ddim_multi':
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            ext_inputs = ext_inputs.repeat_interleave(n_samples, dim=0)
            xs, x0_preds = self.p_sample_loop_ddim(x_masked, ext_inputs)  # 输出denosising_steps个xt以及预测的x0
            # xs, x0_preds = self.p_sample_loop_ddim(x_masked)  # 输出denosising_steps个xt以及预测的x0
            x = xs[-1]  # 最后一个denosising_steps的xt，即x0
            x = x.reshape(B, n_samples, F, V, -1)
        elif self.sample_strategy == 'ddim_one':
            xs, x0_preds = self.p_sample_loop_ddim(x_masked, ext_inputs)
            x = xs[-1].unsqueeze(1)  # 直接取n_samples步数个样本
        if self.sample_strategy == 'ddpm':
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            ext_inputs = ext_inputs.repeat_interleave(n_samples, dim=0)
            x = self.p_sample_loop(x_masked, ext_inputs)
            x = x.reshape(B, n_samples, F, V, -1)
        else:
            raise NotImplementedError
        
        return x, n_params, p, pi
    
    def forward(self, input, ext_inputs, n_samples):

        return self.evaluate(input, ext_inputs, n_samples)

    def loss(self, x0, ext_inputs, c, y_true):
        '''
        loss function
        x0: raw_data without diffusion; shape: (B, F, V, T)
        ext_inputs: external_data_inputs; shape: (B, F, V, T)
        c: x_mask: only know history data; shape: (B, F, V, T)
        '''
        x0      = self.to_diffusion_format(x0).to(self.device)
        c       = c.to(self.device)
        y_true  = y_true.permute(0, 3, 2, 1).to(self.device)

        t = torch.randint(0, self.N, (x0.shape[0],), dtype=torch.long).to(self.device)

        eps = torch.randn_like(x0).to(self.device)

        xt = self.q_xt_x0(x0, t, eps).to(self.device)  # 加入t个时间步的噪声，无需t次迭代
        
        eps_theta, n_params, p, pi = self.eps_model_3(xt, ext_inputs, t, c)  # 降噪过程

        loss1 = F.mse_loss(eps, eps_theta)
        loss2 = stable_zinb_nll_loss(y_true, n_params, p, pi)

        return 0.4 * loss1 + 0.6 * loss2
        

def compute_alpha(beta, t):  # 计算alpha
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


# simple strategy for DDIM
def generalized_steps(x, ext_t, seq, model, b, c):  # equation 18: Sampling Acceleration
    '''
    :params: x -- Initial Gaussian noise， 随机生成的样本序列
    :params: seq -- steps
    :params: model -- denosising model
    :params: b -- beta: variance schedule (represents the noise level)
    :params: c -- condition: x_masked
    '''
    with torch.no_grad():
        # x.size: (batch, nodes, timestep)
        batch = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        # n_params_list = []
        # p_list = []
        # pi_list = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):  # 倒序遍历
            t = (torch.ones(batch) * i).to(x.device)
            t = t.long()
            next_t = (torch.ones(batch) * j).to(x.device)
            at = compute_alpha(b, t)  # 降噪过程当前时刻
            at_next = compute_alpha(b, next_t.long())  # 降噪过程前一个时刻

            xt = xs[-1].to(x.device)
            # print('xt:',xt.shape)
            et, _, _, _ = model(xt, ext_t, t, c)  # the denoising function

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # 根据前向加噪声的公式反推当前时间步的x0
            x0_preds.append(x0_t)  # denosising_steps个x0
            c1 = (
                ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et  # 根据采样策略获得前一个denosising_steps的xt
            # print(c1.shape, c2.shape, at.shape, at_next.shape, x.shape, et.shape, xt_next.shape)
            xs.append(xt_next)  # shape: [-1, V, T+denosising_steps]


        return xs, x0_preds
