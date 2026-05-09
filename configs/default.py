"""
ST-LLM+ 项目配置模块
统一管理所有超参数
"""

import argparse
import sys
import os


def get_args():
    parser = argparse.ArgumentParser(description="ST-LLM+ Traffic Congestion Prediction")

    # ============================================================
    # 数据相关参数
    # ============================================================
    parser.add_argument("--npz_path", type=str,
                        default="/kaggle/input/datasets/chengyy03/biyedatasets/data/ablation_study/full.npz",
                        help="时空数据 npz 文件路径")
    parser.add_argument("--coords_csv_path", type=str,
                        default="/kaggle/input/datasets/chengyy03/nanchang-congest/roadsegid_location.csv",
                        help="路段坐标 CSV 文件路径")
    parser.add_argument("--adj_dist_path", type=str,
                        default="./adj_data/adj_distance.npy",
                        help="物理距离邻接矩阵保存路径")
    parser.add_argument("--adj_sem_path", type=str,
                        default="./adj_data/adj_semantic.npy",
                        help="语义相似邻接矩阵保存路径")
    parser.add_argument("--save_dir", type=str,
                        default="./checkpoints",
                        help="模型保存目录")

    # ============================================================
    # 模型结构参数
    # ============================================================
    parser.add_argument("--num_nodes", type=int, default=500,
                        help="节点数量")
    parser.add_argument("--input_len", type=int, default=12,
                        help="历史输入步长")
    parser.add_argument("--output_len", type=int, default=12,
                        help="未来预测步长")
    parser.add_argument("--feature_dim", type=int, default=8,
                        help="输入特征维度")
    parser.add_argument("--embed_dim", type=int, default=768,
                        help="GPT-2 隐层维度")

    # PFGA 参数
    parser.add_argument("--pfga_num_heads", type=int, default=4,
                        help="PFGA 多头注意力头数")
    parser.add_argument("--pfga_num_layers", type=int, default=2,
                        help="PFGA 层数")
    parser.add_argument("--pfga_dropout", type=float, default=0.1,
                        help="PFGA Dropout 比率")

    # LLM 参数
    parser.add_argument("--llm_model_id", type=str, default="gpt2",
                        help="GPT-2 模型版本 (gpt2 / gpt2-medium / gpt2-large)")
    parser.add_argument("--llm_layer", type=int, default=6,
                        help="保留的 GPT-2 层数")
    parser.add_argument("--llm_unfreeze", type=int, default=3,
                        help="顶端解冻的层数")

    # Output Head 参数
    parser.add_argument("--output_head_hidden", type=int, default=256,
                        help="Output Head MLP 隐层维度")
    parser.add_argument("--output_head_dropout", type=float, default=0.1,
                        help="Output Head Dropout 比率")

    # Temporal Encoder 参数
    parser.add_argument("--temporal_kernel_size", type=int, default=3,
                        help="1D Conv 时间编码器卷积核大小")

    # ============================================================
    # 消融实验控制参数
    # ============================================================
    parser.add_argument("--experiment_name", type=str, default="default",
                        help="消融实验名称")
    parser.add_argument("--use_spatial_pe", action="store_true", default=True,
                        help="是否使用可学习空间位置编码")
    parser.add_argument("--no_spatial_pe", dest="use_spatial_pe", action="store_false",
                        help="消融：去除空间位置编码")
    parser.add_argument("--use_pfga", action="store_true", default=True,
                        help="是否使用 PFGA 图拓扑融合模块")
    parser.add_argument("--no_pfga", dest="use_pfga", action="store_false",
                        help="消融：去除 PFGA 模块")
    parser.add_argument("--use_temporal_conv", action="store_true", default=True,
                        help="是否使用 1D Conv 时间编码器")
    parser.add_argument("--no_temporal_conv", dest="use_temporal_conv", action="store_false",
                        help="消融：替换为简单 Linear 编码器")
    parser.add_argument("--use_dual_graph", action="store_true", default=True,
                        help="是否使用双图融合（距离图+语义图）")
    parser.add_argument("--no_dual_graph", dest="use_dual_graph", action="store_false",
                        help="消融：仅使用距离图")

    # ============================================================
    # 图构建参数
    # ============================================================
    parser.add_argument("--dist_sigma", type=float, default=5.0,
                        help="距离图高斯核 sigma")
    parser.add_argument("--dist_epsilon", type=float, default=0.1,
                        help="距离图稀疏化阈值")
    parser.add_argument("--sem_threshold", type=float, default=0.5,
                        help="语义图相似度阈值")
    parser.add_argument("--poi_features", type=str, default="4,5",
                        help="POI 特征索引 (逗号分隔)")
    parser.add_argument("--ntl_feature", type=int, default=7,
                        help="NTL 特征索引")

    # ============================================================
    # 训练参数
    # ============================================================
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader num_workers，Kaggle 上建议 2~4")
    parser.add_argument("--epochs", type=int, default=50,
                        help="最大训练 epoch 数")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="初始学习率 (Ranger 优化器)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                        help="梯度裁剪最大范数")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Warmup epoch 数")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Early Stopping 耐心值")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # 数据划分
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--stride", type=int, default=1,
                        help="滑窗步长")

    # ============================================================
    # 损失函数参数（六组分多目标正则化）
    # ============================================================
    parser.add_argument("--loss_alpha", type=float, default=1.0,
                        help="Huber Loss 权重（鲁棒基础回归）")
    parser.add_argument("--loss_beta", type=float, default=0.0,
                        help="Quantile Regression Loss 权重（默认关闭，消融实验时可开启）")
    parser.add_argument("--loss_gamma1", type=float, default=0.0,
                        help="Temporal Consistency 权重（默认关闭）")
    parser.add_argument("--loss_gamma2", type=float, default=0.0,
                        help="Curvature Preservation 权重（默认关闭）")
    parser.add_argument("--loss_delta", type=float, default=0.0,
                        help="Variance Preservation 权重（默认关闭）")
    parser.add_argument("--loss_epsilon", type=float, default=0.0,
                        help="Rank Consistency 权重（默认关闭）")
    parser.add_argument("--huber_delta", type=float, default=0.5,
                        help="Huber Loss delta 参数（调低以更早切换至L1）")
    parser.add_argument("--rank_pairs", type=int, default=1024,
                        help="排序一致性损失的采样对数")

    # ============================================================
    # 评估参数
    # ============================================================
    parser.add_argument("--eval_horizons", type=str, default="1,6,12",
                        help="评估的预测步长 (逗号分隔)")
    parser.add_argument("--mape_threshold", type=float, default=0.01,
                        help="MAPE 计算时的最小真实值阈值")
    parser.add_argument("--plot_length", type=int, default=200,
                        help="可视化时展示的时间步数")
    parser.add_argument("--plot_nodes", type=str, default="0,50,100",
                        help="可视化的节点 ID (逗号分隔)")

    # ============================================================
    # 其他
    # ============================================================
    parser.add_argument("--model_path", type=str, default="",
                        help="加载模型路径 (评估时使用)")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="是否使用混合精度训练")

    # Jupyter / Kaggle 兼容
    if "ipykernel" in sys.modules:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    # 解析列表类参数
    args.poi_features = [int(x) for x in args.poi_features.split(",")]
    args.eval_horizons = [int(x) for x in args.eval_horizons.split(",")]
    args.plot_nodes = [int(x) for x in args.plot_nodes.split(",")]

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.adj_dist_path) or ".", exist_ok=True)

    return args
