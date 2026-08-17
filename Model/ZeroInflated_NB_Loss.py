import torch
from torch import nn
import numpy as np
from scipy.stats import nbinom, norm
from scipy.optimize import minimize

def stable_zinb_nll_loss(
    y_true,
    n_params,
    p,
    pi,
    eps=1e-5,
    max_n=1e4,
    reduction="mean",
):
    """
    严格按照截图公式(36)(37)实现的数值稳定 ZINB 负对数似然损失。

    公式参数映射：
        x -> y_true
        r -> n_params
        p -> p
        pi -> pi
    """

    # ZINB损失必须至少使用float32，禁止float16
    y_true = y_true.float()
    n_params = n_params.float()
    p = p.float()
    pi = pi.float()

    # -----------------------------------------------------
    # 输入检查
    # -----------------------------------------------------
    if not torch.isfinite(y_true).all():
        raise FloatingPointError("y_true 中存在 NaN 或 Inf")

    if not torch.isfinite(n_params).all():
        raise FloatingPointError("n_params 中存在 NaN 或 Inf")

    if not torch.isfinite(p).all():
        raise FloatingPointError("p 中存在 NaN 或 Inf")

    if not torch.isfinite(pi).all():
        raise FloatingPointError("pi 中存在 NaN 或 Inf")

    if torch.any(y_true < 0):
        raise ValueError("ZINB标签必须是非负原始计数")

    # -----------------------------------------------------
    # 参数限制
    # -----------------------------------------------------
    n_params = torch.clamp(
        n_params,
        min=eps,
        max=max_n,
    )

    p = torch.clamp(
        p,
        min=eps,
        max=1.0 - eps,
    )

    pi = torch.clamp(
        pi,
        min=eps,
        max=1.0 - eps,
    )

    # 预先计算 log(p), log(1-p), log(pi), log(1-pi)
    log_p = torch.log(p)
    log_one_minus_p = torch.log1p(-p)

    log_pi = torch.log(pi)
    log_one_minus_pi = torch.log1p(-pi)

    # -----------------------------------------------------
    # NB 对数概率
    # log Gamma(r + x) - log Gamma(x + 1) - log Gamma(r) + r * log(1 - p) + x * log(p)
    # -----------------------------------------------------
    log_nb = (
        torch.lgamma(n_params + y_true)
        - torch.lgamma(y_true + 1.0)
        - torch.lgamma(n_params)
        + n_params * log_one_minus_p  # 修正：r * log(1-p)
        + y_true * log_p              # 修正：x * log(p)
    )

    # -----------------------------------------------------
    # x = 0
    # log[ pi + (1 - pi) * (1 - p)^r ]
    # -----------------------------------------------------
    log_prob_zero = torch.logaddexp(
        log_pi,
        log_one_minus_pi + n_params * log_one_minus_p,  # 修正：r * log(1-p)
    )

    # -----------------------------------------------------
    # x > 0
    # log(1 - pi) + log_nb
    # -----------------------------------------------------
    log_prob_positive = (
        log_one_minus_pi
        + log_nb
    )

    log_prob = torch.where(
        y_true == 0,
        log_prob_zero,
        log_prob_positive,
    )

    nll = -log_prob

    if not torch.isfinite(nll).all():
        print("ZINB NLL became non-finite:")
        print("y:", y_true.min().item(), y_true.max().item())
        print("n:", n_params.min().item(), n_params.max().item())
        print("p:", p.min().item(), p.max().item())
        print("pi:", pi.min().item(), pi.max().item())

        raise FloatingPointError("ZINB NLL 中出现 NaN 或 Inf")

    # 对应公式 (37) 中的 1/M * sum(-LL)
    if reduction == "mean":
        return nll.mean()

    if reduction == "sum":
        return nll.sum()

    if reduction == "none":
        return nll

    raise ValueError(f"Unsupported reduction: {reduction}")


def ZINB_Estimation(OD_Sequence, initial_params=None):
    '''
    估计零膨胀负二项分布的参数

    参数：
    - data：样本数据（NumPy数组）
    - initial_params： 包含初始化的参数估计值列表 [zero_prob, n, p]

    return:
    - 估计的零膨胀概率，n， p
    '''

    B, n_samples, F, V, T = OD_Sequence.shape
    OD_Points = OD_Sequence.reshape(-1, n_samples)

    estimated_zero_list = []
    estimated_n_list = []
    estimated_p_list = []

    # 初始化参数
    if initial_params is None:
        zero_inflation_prob = 0.6
        nbinom_n = 4
        nbinom_p = 0.3

    else:
        zero_inflation_prob, nbinom_n, nbinom_p = initial_params

    # 定义似然函数
    def log_likelihood(params, data):
        zero_prob, n, p = params
        log_prob_zero = np.log(zero_prob) * np.sum(data == 0)
        log_prob_non_zero = np.sum(nbinom.logpmf(data[data > 0], n, p))
        total_log_likelihood = log_prob_zero + log_prob_non_zero
        return -total_log_likelihood

    # 最大似然估计
    for i in range(0, OD_Points.shape[0]):
        result = minimize(log_likelihood, np.array([zero_inflation_prob, nbinom_n, nbinom_p]), args=(OD_Points[i], ), method='Nelder-Mead')
        estimated_zero_list.append(result.x[0])
        estimated_n_list.append(result.x[1])
        estimated_p_list.append(result.x[2])

    estimated_zero = np.array(estimated_zero_list)
    estimated_n = np.array(estimated_n_list)
    estimated_p = np.array(estimated_p_list)

    estimated_zero = estimated_zero.reshape((B, -1, F, V, T))
    estimated_n = estimated_n.reshape((B, -1, F, V, T))
    estimated_p = estimated_p.reshape((B, -1, F, V, T))

    # 提取估计的参数
    # estimated_zero_prob, estimated_n, estimated_p = result.x

    return estimated_zero, estimated_n, estimated_p


# data = np.random.randn(48, 10, 1, 1600, 12)
#
# estimated_zero, estimated_n, estimated_p = ZINB_Estimation(data)
#
# print(estimated_zero.shape)

