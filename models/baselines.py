"""
Classic spatio-temporal traffic forecasting baselines.

These implementations keep the same project I/O contract as STLLMModel:
input  (B, T_in, N, C)
output (B, T_out, N, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adj, eps=1e-6):
    """Symmetric adjacency normalization with self loops."""
    adj = adj.float()
    eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    adj = adj + eye
    degree = adj.sum(dim=-1).clamp_min(eps)
    degree_inv_sqrt = degree.pow(-0.5)
    return degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)


class GraphConvolution(nn.Module):
    """A simple first-order graph convolution."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        support = torch.einsum("nm,bmc->bnc", adj, x)
        return self.linear(support)


class SGTCNBaseline(nn.Module):
    """
    Spatial graph convolution plus temporal convolution baseline.

    It first propagates node features through the road graph at each history
    step, then models temporal patterns with 1D convolutions.
    """

    def __init__(self, args):
        super().__init__()
        hidden_dim = getattr(args, "baseline_hidden_dim", 64)
        dropout = getattr(args, "baseline_dropout", 0.1)
        kernel_size = getattr(args, "baseline_kernel_size", 3)
        padding = (kernel_size - 1) // 2

        self.output_len = args.output_len
        self.graph_in = GraphConvolution(args.feature_dim, hidden_dim)
        self.temporal = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), padding=(0, padding)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), padding=(0, padding)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, args.output_len)

    def forward(self, x, adj_dist, adj_sem=None):
        B, T, N, C = x.shape
        adj = normalize_adjacency(adj_dist)

        h = x.reshape(B * T, N, C)
        h = F.relu(self.graph_in(h, adj))
        h = h.reshape(B, T, N, -1).permute(0, 3, 2, 1)
        h = self.temporal(h)
        h = h[:, :, :, -1].permute(0, 2, 1)

        out = self.head(h).permute(0, 2, 1).unsqueeze(-1)
        last_target = x[:, -1:, :, 0:1].expand(-1, self.output_len, -1, -1)
        return out + last_target


class ASTGCNBaseline(nn.Module):
    """
    Attention-based spatio-temporal graph convolution baseline.

    This compact ASTGCN-style model applies temporal attention per node,
    spatial attention per time step, graph propagation, and a horizon head.
    """

    def __init__(self, args):
        super().__init__()
        hidden_dim = getattr(args, "baseline_hidden_dim", 64)
        heads = getattr(args, "baseline_num_heads", 4)
        dropout = getattr(args, "baseline_dropout", 0.1)

        self.output_len = args.output_len
        self.input_proj = nn.Linear(args.feature_dim, hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.spatial_q = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_k = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_v = nn.Linear(hidden_dim, hidden_dim)
        self.graph_conv = GraphConvolution(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, args.output_len),
        )

    def forward(self, x, adj_dist, adj_sem=None):
        B, T, N, C = x.shape
        adj = normalize_adjacency(adj_dist)

        h = self.input_proj(x)

        temporal_in = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        temporal_out, _ = self.temporal_attn(temporal_in, temporal_in, temporal_in)
        temporal_out = self.norm1(temporal_in + self.dropout(temporal_out))
        h = temporal_out.reshape(B, N, T, -1).permute(0, 2, 1, 3)

        node_context = h.mean(dim=1)
        q = self.spatial_q(node_context)
        k = self.spatial_k(node_context)
        spatial_scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        spatial_bias = (adj > 0).float().unsqueeze(0)
        spatial_scores = spatial_scores.masked_fill(spatial_bias == 0, -1e4)
        spatial_weights = torch.softmax(spatial_scores, dim=-1)

        v = self.spatial_v(h)
        spatial_out = torch.einsum("bnm,btmh->btnh", spatial_weights, v)
        spatial_out = self.norm2(h + self.dropout(spatial_out))

        graph_out = F.relu(self.graph_conv(spatial_out.reshape(B * T, N, -1), adj))
        h = graph_out.reshape(B, T, N, -1).mean(dim=1)

        out = self.head(h).permute(0, 2, 1).unsqueeze(-1)
        last_target = x[:, -1:, :, 0:1].expand(-1, self.output_len, -1, -1)
        return out + last_target


class AdaptiveGraphGRUCell(nn.Module):
    """AGCRN-style recurrent cell with an adaptive adjacency matrix."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = GraphConvolution(input_dim + hidden_dim, 2 * hidden_dim)
        self.update = GraphConvolution(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, state, adj):
        combined = torch.cat([x, state], dim=-1)
        gates = torch.sigmoid(self.gate(combined, adj))
        z, r = torch.chunk(gates, chunks=2, dim=-1)
        candidate = torch.cat([x, r * state], dim=-1)
        candidate = torch.tanh(self.update(candidate, adj))
        return z * state + (1.0 - z) * candidate


class AGCRNBaseline(nn.Module):
    """
    Adaptive graph convolutional recurrent baseline.

    Node embeddings learn a data-driven graph, blended with the physical graph
    so the baseline can use both observed topology and adaptive dependencies.
    """

    def __init__(self, args):
        super().__init__()
        hidden_dim = getattr(args, "baseline_hidden_dim", 64)
        embed_dim = getattr(args, "baseline_node_embed_dim", 16)
        dropout = getattr(args, "baseline_dropout", 0.1)

        self.num_nodes = args.num_nodes
        self.output_len = args.output_len
        self.hidden_dim = hidden_dim
        self.node_emb1 = nn.Parameter(torch.randn(args.num_nodes, embed_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(embed_dim, args.num_nodes) * 0.1)
        self.adaptive_alpha = nn.Parameter(torch.tensor(0.5))
        self.cell = AdaptiveGraphGRUCell(args.feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, args.output_len)

    def _adaptive_adj(self, adj_dist):
        learned = F.softmax(F.relu(self.node_emb1 @ self.node_emb2), dim=-1)
        physical = normalize_adjacency(adj_dist)
        alpha = torch.sigmoid(self.adaptive_alpha)
        return alpha * learned + (1.0 - alpha) * physical

    def forward(self, x, adj_dist, adj_sem=None):
        B, T, N, C = x.shape
        adj = self._adaptive_adj(adj_dist)
        state = x.new_zeros(B, N, self.hidden_dim)

        for step in range(T):
            state = self.cell(x[:, step], state, adj)

        state = self.dropout(state)
        out = self.head(state).permute(0, 2, 1).unsqueeze(-1)
        last_target = x[:, -1:, :, 0:1].expand(-1, self.output_len, -1, -1)
        return out + last_target


def build_baseline_model(model_name, args):
    name = model_name.lower()
    if name == "sgtcn":
        return SGTCNBaseline(args)
    if name == "astgcn":
        return ASTGCNBaseline(args)
    if name == "agcrn":
        return AGCRNBaseline(args)
    raise ValueError(f"Unknown baseline model: {model_name}")
