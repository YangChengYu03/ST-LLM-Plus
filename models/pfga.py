"""
PFGA (Prior-knowledge Fused Graph Attention) 模块

改进要点：
1. Softmax 归一化的双图权重门控
2. 残差连接
3. LayerNorm + Dropout
4. 多层堆叠支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PFGALayer(nn.Module):
    """
    单层 PFGA

    融合物理距离图和语义相似图，在节点间进行图感知的注意力信息交互。

    Args:
        embed_dim: 嵌入维度
        num_heads: 多头注意力头数
        dropout: Dropout 比率
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 可学习的双图权重门控 (使用 Softmax 归一化确保和为 1)
        self.graph_gate = nn.Parameter(torch.zeros(2))  # [w_dist_logit, w_sem_logit]

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # LayerNorm + Dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, adj_dist, adj_sem, use_dual_graph=True):
        """
        Args:
            x: (B, N, C) 节点特征
            adj_dist: (N, N) 物理距离邻接矩阵
            adj_sem: (N, N) 语义相似邻接矩阵
            use_dual_graph: 是否使用双图融合（False 时仅用距离图）

        Returns:
            (B, N, C) 更新后的节点特征
        """
        B, N, C = x.shape
        residual = x

        # ======= Pre-Norm + Multi-Head Attention =======
        x_norm = self.norm1(x)

        Q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = self.k_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        V = self.v_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)

        # 注意力分数
        scale = self.head_dim ** 0.5
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, N, N)

        # ======= 图融合（支持消融） =======
        if use_dual_graph:
            # 双图融合：距离图 + 语义图
            gate = F.softmax(self.graph_gate, dim=0)  # 归一化权重
            combined_adj = gate[0] * adj_dist + gate[1] * adj_sem  # (N, N)
        else:
            # 单图消融：仅使用距离图
            combined_adj = adj_dist  # (N, N)

        combined_adj = combined_adj.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)  # (B, H, N, N)

        # 图结构掩码
        mask = (combined_adj <= 0.0)
        attention_scores = attention_scores.masked_fill(mask, -1e4)

        # 加入图先验
        attention_scores = attention_scores + combined_adj

        attn_weights = F.softmax(attention_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        # ======= 残差连接 =======
        x = residual + out

        # ======= Pre-Norm + FFN + 残差连接 =======
        x = x + self.ffn(self.norm2(x))

        return x


class PFGAStack(nn.Module):
    """
    多层 PFGA 堆叠

    Args:
        embed_dim: 嵌入维度
        num_heads: 多头注意力头数
        num_layers: PFGA 层数
        dropout: Dropout 比率
    """

    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PFGALayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, adj_dist, adj_sem, use_dual_graph=True):
        """
        Args:
            x: (B, N, C) 节点特征
            adj_dist: (N, N)
            adj_sem: (N, N)
            use_dual_graph: 是否使用双图融合

        Returns:
            (B, N, C)
        """
        for layer in self.layers:
            x = layer(x, adj_dist, adj_sem, use_dual_graph=use_dual_graph)
        return x
