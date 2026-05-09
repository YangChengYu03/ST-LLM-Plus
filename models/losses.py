"""
自定义损失函数模块

CombinedTrafficLoss: 针对交通拥堵预测的六组分多目标正则化损失函数

设计思路：
- 交通拥堵指数数据高度右偏（大量正常值 ~1.0，少量高拥堵值）
- 纯 MSE/Huber 会被大量正常值主导，导致"均值坍塌"
- 组合六个互补维度的损失项，从回归精度、分布形状、时间一致性、
  二阶曲率保持、方差保持和排序一致性六个方面全面约束模型

总损失:
  L = α·L_Huber + β·L_QR + γ₁·L_TC + γ₂·L_CP + δ·L_VP + ε·L_RC

  L_Huber  : Huber Loss — 鲁棒基础回归 (Huber, 1964)
  L_QR     : Quantile Regression Loss — 分位数感知回归 (Koenker & Bassett, 1978)
  L_TC     : Temporal Consistency Loss — 一阶时间一致性 (余弦相似度)
  L_CP     : Curvature Preservation Loss — 二阶曲率保持正则化 (Bredies et al., 2010; TGV)
  L_VP     : Variance Preservation Loss — 方差保持正则化 (Mathieu et al., 2016)
  L_RC     : Rank Consistency Loss — 排序一致性损失 (Burges et al., 2005)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedTrafficLoss(nn.Module):
    """
    交通拥堵预测六组分多目标正则化损失函数

    Args:
        alpha: Huber Loss 权重（鲁棒基础损失）
        beta: Quantile Regression Loss 权重（分位数感知损失）
        gamma1: Temporal Consistency 权重（一阶时间一致性）
        gamma2: Curvature Preservation 权重（二阶曲率保持正则化）
        delta: Variance Preservation 权重（方差保持正则化）
        epsilon: Rank Consistency 权重（排序一致性损失）
        huber_delta: Huber Loss 的 delta 参数
        quantiles: 分位数列表
        rank_pairs: 排序一致性损失的采样对数
    """

    def __init__(self, alpha=1.0, beta=0.3, gamma1=0.05, gamma2=0.3,
                 delta=0.5, epsilon=0.2, huber_delta=0.5,
                 quantiles=(0.1, 0.5, 0.9), rank_pairs=1024):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.delta = delta
        self.epsilon = epsilon
        self.huber_delta = huber_delta
        self.quantiles = quantiles
        self.rank_pairs = rank_pairs

    def _huber_loss(self, pred, target):
        """
        Huber Loss (Smooth L1 Loss)

        鲁棒回归的基础损失函数 (Huber, 1964)。
        - 误差 < delta 时表现为 L2 Loss（平方惩罚，平滑梯度）
        - 误差 >= delta 时表现为 L1 Loss（线性惩罚，抗异常值）

        delta 设为 0.5（而非默认 1.0），使其更早切换至 L1 行为，
        避免大误差区域的 L2 梯度压制效应进一步加剧均值坍塌。
        """
        return F.smooth_l1_loss(pred, target, beta=self.huber_delta)

    def _quantile_loss(self, pred, target):
        """
        分位数回归损失 (Quantile Regression Loss)

        Koenker & Bassett (1978) 的分位数回归理论。
        通过对 τ={0.1, 0.5, 0.9} 三个分位数联合建模，
        迫使模型不仅学到条件中位数，还关注分布的高低端，打破均值坍塌。

        对于分位数 τ:
            L_τ(e) = max(τ·e, (τ-1)·e)
        """
        losses = []
        errors = target - pred
        for q in self.quantiles:
            losses.append(torch.mean(torch.max(q * errors, (q - 1.0) * errors)))
        return sum(losses) / len(losses)

    def _temporal_consistency_loss(self, pred, target):
        """
        一阶时间一致性损失 (Temporal Consistency Loss)

        使用余弦相似度衡量预测的时间梯度方向与真实值时间梯度方向的一致性。
        相比原始设计中用 MSE 惩罚梯度差异，余弦相似度的关键优势是：
        - 只约束方向一致性，不惩罚幅度
        - 避免压制预测值的变化幅度（MSE 会鼓励平坦输出）
        - 保持模型输出必要的动态变化范围

        pred/target: (B, T_out, N, 1)
        """
        if pred.shape[1] <= 1:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 一阶差分：时间"速度"
        pred_diff = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_diff = target[:, 1:, :, :] - target[:, :-1, :, :]

        # 余弦相似度：只关注方向
        cos_sim = F.cosine_similarity(
            pred_diff.reshape(pred_diff.shape[0], -1),
            target_diff.reshape(target_diff.shape[0], -1),
            dim=-1
        )
        return (1.0 - cos_sim).mean()

    def _curvature_preservation_loss(self, pred, target):
        """
        二阶曲率保持正则化 (Curvature Preservation Loss)

        ★ 解决均值坍塌的关键二阶正则化模块 ★

        理论基础:
        - 交通拥堵峰值在数学上表现为时间序列的高曲率点：
          峰前：一阶导数 > 0 且递增（正曲率/正加速度）
          峰顶：一阶导数 = 0 且曲率极大（曲率变号点）
          峰后：一阶导数 < 0 且递减（负曲率/负加速度）
        - 仅靠一阶约束无法保持这些峰谷的几何形态特征
        - 二阶约束直接迫使模型复现拥堵峰谷的形状

        数学形式:
          设 y(t) 为时间序列，则:
            一阶差分: Δy(t) = y(t+1) - y(t)         # "速度"
            二阶差分: Δ²y(t) = Δy(t+1) - Δy(t)      # "加速度/曲率"
          损失: L_CP = SmoothL1(Δ²ŷ, Δ²y)

        灵感来源:
        - Bredies et al. (2010): Total Generalized Variation (TGV) —
          在图像恢复中引入二阶全变分正则化
        - Papafitsoros & Schönlieb (2014): 一阶与二阶变分的联合方法
        - 信号处理中的二阶差分算子用于检测拐点和曲率变化

        pred/target: (B, T_out, N, 1)
        """
        if pred.shape[1] <= 2:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 一阶差分（速度）
        pred_d1 = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_d1 = target[:, 1:, :, :] - target[:, :-1, :, :]

        # 二阶差分（曲率/加速度）
        pred_d2 = pred_d1[:, 1:, :, :] - pred_d1[:, :-1, :, :]
        target_d2 = target_d1[:, 1:, :, :] - target_d1[:, :-1, :, :]

        # 使用 Smooth L1 (Huber) 而非 MSE，对曲率误差更鲁棒
        return F.smooth_l1_loss(pred_d2, target_d2, beta=self.huber_delta)

    def _variance_preservation_loss(self, pred, target):
        """
        方差保持正则化 (Variance Preservation Loss)

        约束预测值的统计方差接近真实值的方差，从分布层面阻止方差坍塌。

        MSE/Huber 系列损失的全局最优解是条件期望，此时预测方差为零。
        需要额外约束保持输出的分布形态。
        类似于 GAN 中判别器对生成分布的约束作用，但更简单直接。

        灵感来自 Mathieu et al. (2016) "Deep Multi-Scale Video Prediction"
        中的 GDL (Gradient Difference Loss) 相同本质。

        pred/target: (B, T_out, N, 1)
        """
        # 计算每个样本内的方差
        pred_var = torch.var(pred, dim=(1, 2, 3))
        target_var = torch.var(target, dim=(1, 2, 3))
        return F.mse_loss(pred_var, target_var)

    def _rank_consistency_loss(self, pred, target):
        """
        排序一致性损失 (Rank Consistency Loss)

        随机采样样本对，确保预测值的大小排序与真实值一致。
        当 target[a] > target[b] 时，pred[a] 也应 > pred[b]。

        在交通预测中，"哪些路段/时段更拥堵"的相对排序
        比绝对数值预测更有实际决策意义。

        灵感来自 Burges et al. (2005) "Learning to Rank" 中的 pairwise ranking loss。

        pred/target: (B, T_out, N, 1)
        """
        B = pred.shape[0]
        pred_flat = pred.reshape(B, -1)
        target_flat = target.reshape(B, -1)

        num_elements = pred_flat.shape[1]
        num_pairs = min(self.rank_pairs, num_elements)

        # 随机采样索引对
        idx_a = torch.randint(0, num_elements, (num_pairs,), device=pred.device)
        idx_b = torch.randint(0, num_elements, (num_pairs,), device=pred.device)

        target_diff = target_flat[:, idx_a] - target_flat[:, idx_b]
        pred_diff = pred_flat[:, idx_a] - pred_flat[:, idx_b]

        # sign(target_diff) × pred_diff 应为正
        # 当排序一致时 hinge loss 为零
        sign_target = torch.sign(target_diff)
        return F.relu(-sign_target * pred_diff).mean()

    def forward(self, pred, target):
        """
        Args:
            pred: (B, T_out, N, 1) 模型预测值
            target: (B, T_out, N, 1) 真实值

        Returns:
            total_loss: 标量
            loss_dict: 各分量的字典（用于日志记录）
        """
        # 六个损失组分
        loss_huber = self._huber_loss(pred, target)
        loss_quantile = self._quantile_loss(pred, target)
        loss_tc = self._temporal_consistency_loss(pred, target)
        loss_cp = self._curvature_preservation_loss(pred, target)
        loss_vp = self._variance_preservation_loss(pred, target)
        loss_rc = self._rank_consistency_loss(pred, target)

        total_loss = (self.alpha * loss_huber
                      + self.beta * loss_quantile
                      + self.gamma1 * loss_tc
                      + self.gamma2 * loss_cp
                      + self.delta * loss_vp
                      + self.epsilon * loss_rc)

        loss_dict = {
            'total': total_loss.item(),
            'huber': loss_huber.item(),
            'quantile': loss_quantile.item(),
            'tc': loss_tc.item(),
            'cp': loss_cp.item(),
            'vp': loss_vp.item(),
            'rc': loss_rc.item(),
        }

        return total_loss, loss_dict
