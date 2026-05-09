"""
ST-LLM+ 主模型

改进要点：
1. 1D Conv 时间维度编码（取代简单展平）
2. 可学习的空间位置编码
3. PFGA 输出后的 LayerNorm + Dropout
4. MLP Output Head（取代单线性层）
5. LoRA 微调 GPT-2（取代直接解冻，防止过拟合）
6. 图结构注入 GPT-2 Attention Mask（拓扑约束）
7. 消融实验支持：可选关闭各模块
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from models.pfga import PFGAStack
from dataclasses import dataclass
from typing import Optional, Tuple


# ==========================================
# 自定义输出容器（用于 custom_forward）
# ==========================================

@dataclass
class BaseModelOutput:
    last_hidden_state: torch.FloatTensor = None


# ==========================================
# 时间编码器
# ==========================================

class TemporalEncoder(nn.Module):
    """
    1D 卷积时间编码器

    将 (B, N, T_in, F) 的时序特征通过 1D 卷积投影到 embed_dim。
    比简单展平更好地保留时间维度的局部模式。
    """

    def __init__(self, input_len, feature_dim, embed_dim, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(feature_dim, embed_dim // 2, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

        # 自适应池化将时间维度压缩为 1
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: (B, N, T, F) 时空特征

        Returns:
            (B, N, embed_dim) 节点嵌入
        """
        B, N, T, F = x.shape

        # 重塑为 (B*N, F, T) 以适配 Conv1d
        x = x.reshape(B * N, T, F).permute(0, 2, 1)  # (B*N, F, T)

        x = self.act(self.conv1(x))  # (B*N, embed_dim//2, T)
        x = self.act(self.conv2(x))  # (B*N, embed_dim, T)
        x = self.pool(x).squeeze(-1)  # (B*N, embed_dim)
        x = x.reshape(B, N, -1)      # (B, N, embed_dim)
        x = self.norm(x)

        return x


class LinearTemporalEncoder(nn.Module):
    """
    简单线性时间编码器（消融实验用）

    直接将 (T_in * F) 展平后通过线性投影到 embed_dim。
    用于对比 1D Conv 时间编码器的效果。
    """

    def __init__(self, input_len, feature_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_len * feature_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, N, T, F) 时空特征

        Returns:
            (B, N, embed_dim) 节点嵌入
        """
        B, N, T, F = x.shape
        x = x.reshape(B, N, T * F)  # (B, N, T*F)
        x = self.proj(x)            # (B, N, embed_dim)
        x = self.norm(x)
        return x


class OutputHead(nn.Module):
    """
    MLP 输出头

    将 GPT-2 提取的特征映射为未来 T_out 步的预测值。
    使用两层 MLP 替代单线性层，增强非线性表达能力。
    """

    def __init__(self, embed_dim, output_len, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.output_len = output_len
        self.horizon_embedding = nn.Parameter(torch.randn(output_len, embed_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, last_target=None):
        """
        Args:
            x: (B, N, embed_dim)
            last_target: (B, N, 1) or None

        Returns:
            (B, T_out, N, 1)
        """
        B, N, C = x.shape
        horizon_context = self.horizon_embedding.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        hidden_context = x.unsqueeze(2).expand(-1, -1, self.output_len, -1)
        fused = torch.cat([hidden_context, horizon_context], dim=-1)

        out = self.mlp(fused).permute(0, 2, 1, 3)

        if last_target is not None:
            baseline = last_target.unsqueeze(1).expand(-1, self.output_len, -1, -1)
            out = out + baseline

        return out


# ==========================================
# ST-LLM+ 主模型
# ==========================================

class STLLMModel(nn.Module):
    """
    ST-LLM+ 总体模型

    Pipeline:
    1. TemporalEncoder: (B, T_in, N, F) -> (B, N, embed_dim) — 时间维度编码
    2. + SpatialPositionEncoding — 可学习的空间位置编码
    3. PFGAStack: (B, N, embed_dim) -> (B, N, embed_dim) — 图拓扑融合
    4. LayerNorm + Dropout — 过渡层
    5. GPT-2 + LoRA + Graph Attention Mask — 时空演变建模
    6. OutputHead: (B, N, embed_dim) -> (B, T_out, N, 1) — 预测输出

    消融控制:
    - use_spatial_pe: 是否使用空间位置编码
    - use_pfga: 是否使用 PFGA 图拓扑融合
    - use_temporal_conv: 是否使用 1D Conv 时间编码器
    - use_dual_graph: 是否使用双图融合（传递到 PFGA）
    """

    def __init__(self, args):
        super().__init__()
        embed_dim = args.embed_dim
        self.embed_dim = embed_dim

        # 消融控制标志
        self.use_spatial_pe = getattr(args, 'use_spatial_pe', True)
        self.use_pfga = getattr(args, 'use_pfga', True)
        self.use_temporal_conv = getattr(args, 'use_temporal_conv', True)
        self.use_dual_graph = getattr(args, 'use_dual_graph', True)

        # 1. 时间维度编码器（可选 Conv 或 Linear）
        if self.use_temporal_conv:
            self.temporal_encoder = TemporalEncoder(
                input_len=args.input_len,
                feature_dim=args.feature_dim,
                embed_dim=embed_dim,
                kernel_size=args.temporal_kernel_size,
            )
        else:
            self.temporal_encoder = LinearTemporalEncoder(
                input_len=args.input_len,
                feature_dim=args.feature_dim,
                embed_dim=embed_dim,
            )

        # 2. 可学习的空间位置编码（可选）
        if self.use_spatial_pe:
            self.spatial_pos_enc = nn.Parameter(
                torch.randn(1, args.num_nodes, embed_dim) * 0.02
            )

        # 3. PFGA 图拓扑融合（可选）
        if self.use_pfga:
            self.pfga = PFGAStack(
                embed_dim=embed_dim,
                num_heads=args.pfga_num_heads,
                num_layers=args.pfga_num_layers,
                dropout=args.pfga_dropout,
            )

        # 4. 过渡层
        self.pre_llm_norm = nn.LayerNorm(embed_dim)
        self.pre_llm_dropout = nn.Dropout(args.pfga_dropout)

        # 5. GPT-2 基座（含 LoRA）
        self.gpt_layers = args.llm_layer
        self.llm_model, self.n_head = self._build_llm(args)

        # 6. 输出头
        self.output_head = OutputHead(
            embed_dim=embed_dim,
            output_len=args.output_len,
            hidden_dim=args.output_head_hidden,
            dropout=args.output_head_dropout,
        )

        # Dropout for GPT output
        self.gpt_dropout = nn.Dropout(0.1)

    def _build_llm(self, args):
        """
        加载、截断 GPT-2，应用 LoRA 微调策略

        LoRA 策略（恢复原版设计）：
        - 对 GPT-2 的 c_attn 层应用低秩适配 (r=16, alpha=32)
        - 仅训练 LoRA 参数 + LayerNorm + 位置编码
        - 冻结全部原始 GPT-2 参数，防止大规模参数空间导致过拟合
        """
        print(f"正在加载 GPT-2 基座模型: {args.llm_model_id} ...")
        llm = AutoModel.from_pretrained(args.llm_model_id)
        n_head = llm.config.n_head

        # 截断到指定层数
        if args.llm_layer < len(llm.h):
            llm.h = llm.h[:args.llm_layer]
            print(f"  ✂️ 已将 GPT-2 截断至前 {args.llm_layer} 层")

        U = args.llm_unfreeze  # 顶端解冻层数

        # ======= 应用 LoRA =======
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['c_attn'],
            bias="none",
        )
        llm = get_peft_model(llm, lora_config)
        print(f"  🔧 LoRA 已应用 (r=16, alpha=32, target=c_attn)")
        llm.print_trainable_parameters()

        # ======= 精细化冻结策略（恢复原版逻辑） =======
        # 底层 (0 ~ gpt_layers-U-1): 冻结大部分，仅保留 ln / wpe 可训练
        # 顶层 (gpt_layers-U ~ gpt_layers-1): 冻结 mlp，其余可训练
        freeze_layers = args.llm_layer - U

        for layer_index, layer in enumerate(llm.base_model.model.h):
            for name, param in layer.named_parameters():
                if layer_index < freeze_layers:
                    # 底层：仅 ln 和 wpe 可训练
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    elif "lora" in name:
                        param.requires_grad = True  # LoRA 参数始终可训练
                    else:
                        param.requires_grad = False
                else:
                    # 顶层：mlp 冻结，其余（attention + ln + lora）可训练
                    if "mlp" in name and "lora" not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        # 冻结 wte (token embedding，我们不需要因为输入不是文本)
        for param in llm.base_model.model.wte.parameters():
            param.requires_grad = False

        # 保持 wpe (位置编码) 可训练 — 与原版一致
        for param in llm.base_model.model.wpe.parameters():
            param.requires_grad = True

        print(f"  ❄️ 冻结策略: 底层{freeze_layers}层仅ln/lora可训练，顶层{U}层冻结mlp")
        print(f"  🔥 wpe 保持可训练")

        return llm, n_head

    def _build_gpt_attention_mask(self, adj_dist, batch_size, device, dtype):
        """
        将图邻接矩阵转换为 GPT-2 的 Attention Mask

        未连接的节点对将被赋予 -10000.0 的掩码值，
        使 GPT-2 的注意力层只关注图中有连接的节点。

        Args:
            adj_dist: (N, N) 距离邻接矩阵
            batch_size: batch 大小
            device: 设备
            dtype: 数据类型

        Returns:
            (B, n_head, N, N) 注意力掩码
        """
        # (1, 1, N, N) -> (B, n_head, N, N)
        mask = adj_dist.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, self.n_head, -1, -1)
        # 不连接的位置设为极大负值，连接的位置设为 0
        attention_mask = (1.0 - mask) * -10000.0
        return attention_mask.to(device=device, dtype=dtype)

    def _custom_gpt_forward(self, inputs_embeds, attention_mask=None):
        """
        自定义 GPT-2 前向传播（绕过标准 forward 以支持 4D attention mask）

        与原版 PFA.custom_forward 逻辑一致：
        - 手动计算位置编码
        - 逐层前向传播
        - 支持自定义 4D attention mask

        Args:
            inputs_embeds: (B, N, embed_dim)
            attention_mask: (B, n_head, N, N) 或 None

        Returns:
            BaseModelOutput
        """
        # 获取 GPT-2 基座（穿透 LoRA 包装）
        gpt2 = self.llm_model.base_model.model

        input_shape = inputs_embeds.size()[:-1]
        device = inputs_embeds.device

        # 计算位置编码
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape[0], -1)
        position_embeds = gpt2.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.gpt_dropout(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        # 逐层前向传播
        for block in gpt2.h:
            outputs = block(
                hidden_states,
                layer_past=None,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = outputs[0]

        # 最终 LayerNorm
        hidden_states = gpt2.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        return BaseModelOutput(last_hidden_state=hidden_states)

    def forward(self, x, adj_dist, adj_sem):
        """
        Args:
            x: (B, T_in, N, F) 输入时空特征
            adj_dist: (N, N) 物理距离邻接矩阵
            adj_sem: (N, N) 语义邻接矩阵

        Returns:
            (B, T_out, N, 1) 预测值
        """
        B, T, N, F = x.shape

        # Step 1: 时间维度编码
        #   (B, T, N, F) -> (B, N, T, F) -> (B, N, embed_dim)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)
        node_embeds = self.temporal_encoder(x)  # (B, N, embed_dim)

        # Step 2: 添加空间位置编码（可选）
        if self.use_spatial_pe:
            node_embeds = node_embeds + self.spatial_pos_enc

        # Step 3: PFGA 图拓扑融合（可选）
        if self.use_pfga:
            spatial_embeds = self.pfga(
                node_embeds, adj_dist, adj_sem,
                use_dual_graph=self.use_dual_graph
            )
        else:
            spatial_embeds = node_embeds

        # Step 4: 过渡层
        spatial_embeds = self.pre_llm_dropout(self.pre_llm_norm(spatial_embeds))

        # Step 5: GPT-2 + 图结构 Attention Mask
        # 构建基于邻接矩阵的 attention mask，注入图拓扑约束
        attn_mask = self._build_gpt_attention_mask(
            adj_dist, B, spatial_embeds.device, spatial_embeds.dtype
        )

        # 使用自定义 forward 以支持 4D attention mask
        llm_outputs = self._custom_gpt_forward(
            inputs_embeds=spatial_embeds,
            attention_mask=attn_mask,
        )
        hidden_states = llm_outputs.last_hidden_state  # (B, N, embed_dim)
        hidden_states = self.gpt_dropout(hidden_states)

        # Step 6: 预测输出
        last_target = x[:, :, -1, 0:1]
        predictions = self.output_head(hidden_states, last_target=last_target)  # (B, T_out, N, 1)

        return predictions

    def get_trainable_params(self):
        """获取所有需要训练的参数"""
        return [p for p in self.parameters() if p.requires_grad]

    def ensure_float32_trainable(self):
        """将所有可训练参数强制转为 FP32（混合精度训练需要）"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
