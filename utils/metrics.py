import torch
import math
import numpy as np

# =========================================================
# 1. 反归一化：统一输出到 CPU
# =========================================================

def inverse_normalize_to_cpu(clean_data, x):
    """
    将 Tensor 反归一化到原始客流尺度，并统一返回 CPU Tensor。

    x 可以是：
        [B, S, F, V, T]
        [B, F, V, T]
        [B, V, T]
    """

    x_cpu = x.detach().cpu()

    try:
        result = clean_data.reverse_normalization(x_cpu)
    except Exception:
        result = clean_data.reverse_normalization(
            x_cpu.numpy()
        )

    if isinstance(result, np.ndarray):
        result = torch.from_numpy(result)

    elif not torch.is_tensor(result):
        result = torch.as_tensor(result)

    return result.float()


# =========================================================
# 2. 从已排序样本计算任意分位数
# =========================================================

def quantile_from_sorted(sorted_samples, q):
    """
    sorted_samples:
        [B, S, V, T]，样本维 S 已按升序排序。

    q:
        分位数，例如 0.025 或 0.975。

    return:
        [B, V, T]
    """

    sample_num = sorted_samples.shape[1]

    if sample_num < 2:
        return sorted_samples[:, 0]

    position = q * (sample_num - 1)

    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    weight = position - lower_index

    lower_value = sorted_samples[:, lower_index]
    upper_value = sorted_samples[:, upper_index]

    return (
        lower_value * (1.0 - weight)
        + upper_value * weight
    )


# =========================================================
# 3. 扩散样本指标
# =========================================================

def calculate_sample_distribution_metrics(
    samples,
    target,
):
    """
    根据扩散样本计算预测均值、95%区间和 CRPS。

    Parameters
    ----------
    samples:
        [B, S, V, T]，原始客流尺度。

    target:
        [B, V, T]，原始客流尺度。

    Returns
    -------
    prediction_mean:
        [B,V,T]

    lower95:
        [B,V,T]

    upper95:
        [B,V,T]

    crps:
        [B,V,T]
    """

    samples = samples.float()
    target = target.float()

    if samples.ndim != 4:
        raise ValueError(
            "samples 应为 [B,S,V,T]，"
            f"当前为 {tuple(samples.shape)}"
        )

    if target.ndim != 3:
        raise ValueError(
            "target 应为 [B,V,T]，"
            f"当前为 {tuple(target.shape)}"
        )

    if samples.shape[0] != target.shape[0]:
        raise ValueError("samples 与 target 的 batch 不一致。")

    if samples.shape[2:] != target.shape[1:]:
        raise ValueError(
            "samples 与 target 的节点/时间维不一致："
            f"{tuple(samples.shape)} vs {tuple(target.shape)}"
        )

    sample_num = samples.shape[1]

    # -----------------------------------------------------
    # 点预测：扩散样本均值
    # -----------------------------------------------------
    prediction_mean = samples.mean(dim=1)

    # -----------------------------------------------------
    # 样本排序，只排序一次
    # 后面区间和 CRPS 共用
    # -----------------------------------------------------
    sorted_samples = torch.sort(
        samples,
        dim=1,
    ).values

    # -----------------------------------------------------
    # 95%预测区间
    # -----------------------------------------------------
    lower95 = quantile_from_sorted(
        sorted_samples,
        q=0.025,
    )

    upper95 = quantile_from_sorted(
        sorted_samples,
        q=0.975,
    )

    lower95 = torch.clamp(lower95, min=0.0)
    upper95 = torch.clamp(upper95, min=0.0)

    # -----------------------------------------------------
    # 经验 CRPS
    #
    # CRPS =
    # E|X-y| - 0.5 E|X-X'|
    # -----------------------------------------------------

    # 第一项：E|X-y|
    term1 = torch.mean(
        torch.abs(
            samples - target.unsqueeze(1)
        ),
        dim=1,
    )

    # 第二项：
    # 0.5 * E|X-X'|
    coefficients = (
        2.0
        * torch.arange(
            1,
            sample_num + 1,
            dtype=sorted_samples.dtype,
            device=sorted_samples.device,
        )
        - sample_num
        - 1.0
    )

    coefficients = coefficients.view(
        1,
        sample_num,
        1,
        1,
    )

    term2 = torch.sum(
        coefficients * sorted_samples,
        dim=1,
    ) / float(sample_num ** 2)

    crps = term1 - term2
    crps = torch.clamp(crps, min=0.0)

    return (
        prediction_mean,
        lower95,
        upper95,
        crps,
    )
