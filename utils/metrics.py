"""
评估指标模块

封装 MAE, RMSE, Masked MAPE, WAPE 计算。
支持分步长评估 (per-horizon)。
"""

import numpy as np


def masked_mae(pred, target, threshold=0.0):
    """平均绝对误差"""
    return np.mean(np.abs(pred - target))


def masked_rmse(pred, target, threshold=0.0):
    """均方根误差"""
    return np.sqrt(np.mean((pred - target) ** 2))


def masked_mape(pred, target, threshold=0.01):
    """
    Masked MAPE

    忽略真实值低于 threshold 的数据点，изб免分母接近零导致 MAPE 爆炸。
    """
    mask = np.abs(target) > threshold
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100


def wape(pred, target):
    """
    WAPE (Weighted Absolute Percentage Error)

    比 MAPE 更鲁棒：
    WAPE = sum(|pred - target|) / sum(|target|) × 100
    """
    total_abs_error = np.sum(np.abs(pred - target))
    total_abs_target = np.sum(np.abs(target))
    if total_abs_target == 0:
        return 0.0
    return (total_abs_error / total_abs_target) * 100


def compute_metrics(pred, target, mape_threshold=0.01):
    """
    计算全部评估指标

    Args:
        pred: numpy array, 预测值
        target: numpy array, 真实值
        mape_threshold: MAPE 最小真实值阈值

    Returns:
        dict: {mae, rmse, mape, wape}
    """
    return {
        'mae': float(masked_mae(pred, target)),
        'rmse': float(masked_rmse(pred, target)),
        'mape': float(masked_mape(pred, target, threshold=mape_threshold)),
        'wape': float(wape(pred, target)),
    }


def compute_per_horizon_metrics(pred, target, horizons, mape_threshold=0.01):
    """
    分步长评估指标

    Args:
        pred: numpy array, (num_samples, T_out, N) 或 (num_samples, T_out, N, 1)
        target: numpy array, 同 pred
        horizons: list of int, 要评估的预测步长（1-based index）
        mape_threshold: MAPE 最小真实值阈值

    Returns:
        dict: {horizon: {mae, rmse, mape, wape}, ...}
    """
    results = {}
    for h in horizons:
        idx = h - 1  # 转为 0-based
        if idx >= pred.shape[1]:
            continue
        p = pred[:, idx]
        t = target[:, idx]
        results[f"horizon_{h}"] = compute_metrics(p, t, mape_threshold)
    return results


def format_metrics(metrics_dict, prefix=""):
    """格式化指标字典为可打印字符串"""
    parts = []
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            sub = format_metrics(value, prefix=f"{key}/")
            parts.append(sub)
        else:
            parts.append(f"{prefix}{key}: {value:.4f}")
    return " | ".join(parts)
