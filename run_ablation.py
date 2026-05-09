"""
ST-LLM+ 消融实验批量运行器

功能：
1. 读取 YAML 配置文件中的实验定义
2. 支持按 batch 分批运行
3. 断点续跑（跳过已完成的实验）
4. 自动训练 + 评估 + 保存结果
5. 运行结束后生成汇总对比表

使用方式：
    python run_ablation.py                            # 运行所有 enabled 实验
    python run_ablation.py --batch 1                  # 只运行 batch 1
    python run_ablation.py --batch 2                  # 只运行 batch 2
    python run_ablation.py --only feat_full            # 只运行指定实验
    python run_ablation.py --skip-completed            # 跳过已有结果的实验
    python run_ablation.py --config path/to/config.yaml  # 自定义配置文件
"""

import os
import sys
import json
import time
import copy
import random
import argparse
import numpy as np
import torch

from torch.amp import autocast, GradScaler

# 兼容 YAML（如果没有 pyyaml 就用 json fallback）
try:
    import yaml
except ImportError:
    print("⚠️ pyyaml 未安装，尝试 pip install pyyaml ...")
    os.system(f"{sys.executable} -m pip install pyyaml -q")
    import yaml

from configs.default import get_args
from data.graph_builder import build_dist_graph, build_semantic_graph
from data.dataset import create_dataloaders
from models.st_llm import STLLMModel
from models.losses import CombinedTrafficLoss
from utils.logger import TrainLogger, EarlyStopping
from utils.metrics import compute_metrics, compute_per_horizon_metrics, wape
from ranger21 import Ranger


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed):
    """设置随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_real_mae(preds, targets, scaler):
    """
    计算反归一化后的真实值 MAE

    确保训练/验证/测试三个阶段使用相同口径的指标。
    """
    with torch.no_grad():
        preds_real = scaler.inverse_transform(preds.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
        targets_real = scaler.inverse_transform(targets.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
        mae = np.mean(np.abs(preds_real - targets_real))
    return mae


def compute_real_wape(preds, targets, scaler):
    """计算反归一化后的真实值 WAPE"""
    with torch.no_grad():
        preds_real = scaler.inverse_transform(preds.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
        targets_real = scaler.inverse_transform(targets.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
        metric = wape(preds_real, targets_real)
    return metric


def compute_real_metrics(preds, targets, scaler, mape_threshold=0.01):
    """Compute MAE/RMSE/MAPE/WAPE after inverse transform."""
    with torch.no_grad():
        preds_real = scaler.inverse_transform(preds.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
        targets_real = scaler.inverse_transform(targets.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
        metrics = compute_metrics(preds_real, targets_real, mape_threshold=mape_threshold)
    return {
        'mae': float(metrics['mae']),
        'rmse': float(metrics['rmse']),
        'mape': float(metrics['mape']),
        'wape': float(metrics['wape']),
    }


def load_ablation_config(config_path):
    """加载消融实验 YAML 配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_args(base_args, defaults, experiment_overrides):
    """
    合并参数：base_args (argparse defaults) <- defaults (yaml) <- experiment (yaml)

    返回合并后的 args 对象
    """
    args = copy.deepcopy(base_args)

    # 应用 YAML defaults
    for key, value in defaults.items():
        if key not in ('npz_base_dir', 'results_base_dir', 'coords_csv_path'):
            if hasattr(args, key):
                setattr(args, key, value)

    # 应用实验覆盖参数
    for key, value in experiment_overrides.items():
        if key in ('enabled', 'batch', 'npz_file', 'description'):
            continue  # 这些是元数据，不是模型参数
        if hasattr(args, key):
            setattr(args, key, value)

    # 处理 npz_path
    npz_base = defaults.get('npz_base_dir', '')
    npz_file = experiment_overrides.get('npz_file', 'full.npz')
    args.npz_path = os.path.join(npz_base, npz_file)

    # 处理 coords_csv_path
    if 'coords_csv_path' in defaults:
        args.coords_csv_path = defaults['coords_csv_path']

    return args


def is_experiment_completed(result_dir):
    """检查实验是否已完成（results.json 存在）"""
    return os.path.exists(os.path.join(result_dir, 'results.json'))


# ============================================================
# 训练核心逻辑
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, scaler_amp,
                    A_dist, A_sem, device, max_grad_norm, use_amp, scaler, mape_threshold=0.01):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_real_mae = 0.0
    total_real_rmse = 0.0
    total_real_mape = 0.0
    total_real_wape = 0.0
    loss_components = {'huber': 0.0, 'quantile': 0.0, 'tc': 0.0, 'cp': 0.0, 'vp': 0.0, 'rc': 0.0}
    num_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16, enabled=use_amp):
            preds = model(batch_x, A_dist, A_sem)
            loss, loss_dict = criterion(preds, batch_y)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss_dict['total']
        for k in loss_components:
            loss_components[k] += loss_dict.get(k, 0.0)
        real_metrics = compute_real_metrics(preds, batch_y, scaler, mape_threshold=mape_threshold)
        total_real_mae += real_metrics['mae']
        total_real_rmse += real_metrics['rmse']
        total_real_mape += real_metrics['mape']
        total_real_wape += real_metrics['wape']
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    train_metrics = {
        'mae': total_real_mae / max(num_batches, 1),
        'rmse': total_real_rmse / max(num_batches, 1),
        'mape': total_real_mape / max(num_batches, 1),
        'wape': total_real_wape / max(num_batches, 1),
    }
    for k in loss_components:
        loss_components[k] /= max(num_batches, 1)

    return avg_loss, train_metrics, loss_components


@torch.no_grad()
def validate(model, val_loader, criterion, A_dist, A_sem, device, use_amp, scaler, mape_threshold=0.01):
    """验证一个 epoch，同时返回归一化损失和真实值指标。"""
    model.eval()
    total_loss = 0.0
    preds_real_all = []
    trues_real_all = []
    num_batches = 0

    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        with autocast('cuda', dtype=torch.float16, enabled=use_amp):
            preds = model(batch_x, A_dist, A_sem)
            loss, loss_dict = criterion(preds, batch_y)

        total_loss += loss_dict['total']

        preds_real = scaler.inverse_transform(preds.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
        trues_real = scaler.inverse_transform(batch_y.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
        preds_real_all.append(preds_real)
        trues_real_all.append(trues_real)
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    if preds_real_all:
        preds_real_all = np.concatenate(preds_real_all, axis=0)
        trues_real_all = np.concatenate(trues_real_all, axis=0)
        metrics = compute_metrics(preds_real_all, trues_real_all, mape_threshold=mape_threshold)
    else:
        metrics = {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'wape': 0.0}

    return avg_loss, metrics


@torch.no_grad()
def evaluate_model(model, test_loader, A_dist, A_sem, scaler, device, args):
    """在测试集上评估模型，返回指标字典"""
    model.eval()
    all_preds = []
    all_trues = []

    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)

        with autocast('cuda', dtype=torch.float16, enabled=args.use_amp):
            preds = model(batch_x, A_dist, A_sem)

        all_preds.append(preds.float().cpu().numpy())
        all_trues.append(batch_y.float().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze(-1)  # (S, T_out, N)
    all_trues = np.concatenate(all_trues, axis=0).squeeze(-1)  # (S, T_out, N)

    # 反归一化
    preds_inv = scaler.inverse_transform(all_preds, feature_idx=0)
    trues_inv = scaler.inverse_transform(all_trues, feature_idx=0)

    # 整体指标
    overall = compute_metrics(preds_inv, trues_inv, mape_threshold=args.mape_threshold)

    # 分步长指标
    per_horizon = compute_per_horizon_metrics(
        preds_inv, trues_inv,
        horizons=args.eval_horizons,
        mape_threshold=args.mape_threshold
    )

    return {
        'overall': overall,
        'per_horizon': per_horizon,
    }


# ============================================================
# 单个实验运行器
# ============================================================

def run_single_experiment(exp_name, args, result_dir, device):
    """
    运行单个消融实验：训练 + 评估 + 保存结果

    Args:
        exp_name: 实验名称
        args: 合并后的配置参数
        result_dir: 结果保存目录
        device: 训练设备

    Returns:
        results_dict: 实验结果
    """
    os.makedirs(result_dir, exist_ok=True)
    set_seed(args.seed)

    exp_start_time = time.time()

    # ------ 1. 构建邻接矩阵 ------
    adj_dir = os.path.join(result_dir, "adj_data")
    os.makedirs(adj_dir, exist_ok=True)
    adj_dist_path = os.path.join(adj_dir, "adj_distance.npy")
    adj_sem_path = os.path.join(adj_dir, "adj_semantic.npy")

    # 更新 args 中的邻接矩阵路径
    args.adj_dist_path = adj_dist_path
    args.adj_sem_path = adj_sem_path

    if not os.path.exists(adj_dist_path):
        print("  📐 构建物理距离图...")
        build_dist_graph(
            npz_path=args.npz_path,
            coords_csv_path=args.coords_csv_path,
            save_path=adj_dist_path,
            sigma=args.dist_sigma,
            epsilon=args.dist_epsilon,
        )

    if not os.path.exists(adj_sem_path):
        print("  📐 构建语义相似图...")
        build_semantic_graph(
            npz_path=args.npz_path,
            save_path=adj_sem_path,
            poi_features=args.poi_features,
            ntl_feature=args.ntl_feature,
            threshold=args.sem_threshold,
        )

    A_dist = torch.FloatTensor(np.load(adj_dist_path)).to(device)
    A_sem = torch.FloatTensor(np.load(adj_sem_path)).to(device)

    # ------ 2. 数据加载 ------
    train_loader, val_loader, test_loader, scaler = create_dataloaders(args)

    # ------ 3. 构建模型 ------
    print(f"\n  🏗️ 构建模型 (实验: {exp_name})...")
    print(f"     use_spatial_pe:    {args.use_spatial_pe}")
    print(f"     use_pfga:          {args.use_pfga}")
    print(f"     use_temporal_conv: {args.use_temporal_conv}")
    print(f"     use_dual_graph:    {args.use_dual_graph}")

    model = STLLMModel(args).to(device)
    model.ensure_float32_trainable()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     总参数量:   {num_params:,}")
    print(f"     可训练参数: {num_trainable:,}")

    # ------ 4. 损失函数、优化器 ------
    criterion = CombinedTrafficLoss(
        alpha=args.loss_alpha,
        beta=args.loss_beta,
        gamma1=args.loss_gamma1,
        gamma2=args.loss_gamma2,
        delta=args.loss_delta,
        epsilon=args.loss_epsilon,
        huber_delta=args.huber_delta,
        rank_pairs=args.rank_pairs,
    )

    optimizer = Ranger(
        model.get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler_amp = GradScaler('cuda', enabled=args.use_amp)

    # ------ 5. 训练循环 ------
    logger = TrainLogger(log_dir=result_dir)
    early_stopping = EarlyStopping(patience=args.early_stop_patience)
    best_model_path = os.path.join(result_dir, "best_model.pth")

    print(f"\n  🚀 开始训练 (最多 {args.epochs} 个 epoch)...")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_metrics, train_components = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler_amp,
            A_dist, A_sem, device, args.max_grad_norm, args.use_amp, scaler,
            mape_threshold=args.mape_threshold
        )
        train_mae = train_metrics['mae']
        train_rmse = train_metrics['rmse']
        train_mape = train_metrics['mape']
        train_wape = train_metrics['wape']

        val_loss, val_metrics = validate(
            model, val_loader, criterion, A_dist, A_sem, device, args.use_amp, scaler,
            mape_threshold=args.mape_threshold
        )
        val_mae = val_metrics['mae']
        val_rmse = val_metrics['rmse']
        val_mape = val_metrics['mape']
        val_wape = val_metrics['wape']

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        print(f"  Epoch {epoch:03d}/{args.epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Train MAE(real): {train_mae:.4f} | "
              f"Train RMSE(real): {train_rmse:.4f} | "
              f"Train MAPE(real): {train_mape:.2f}% | "
              f"Val MAE(real): {val_mae:.4f} | "
              f"Val RMSE(real): {val_rmse:.4f} | "
              f"Val MAPE(real): {val_mape:.2f}% | "
              f"Train WAPE(real): {train_wape:.2f}% | "
              f"Val WAPE(real): {val_wape:.2f}% | "
              f"LR: {current_lr:.6f}")

        logger.log_epoch(
            epoch,
            train_loss,
            val_loss,
            current_lr,
            epoch_time,
            extra_metrics={
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_mape': train_mape,
                'train_wape': train_wape,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_mape': val_mape,
                'val_wape': val_wape,
            },
        )

        # Early Stopping 基于真实值 MAE（与测试集评估对齐）
        is_best = early_stopping(val_mae)
        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_mape': train_mape,
                'train_wape': train_wape,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_mape': val_mape,
                'val_wape': val_wape,
                'experiment_name': exp_name,
            }, best_model_path)
            print(
                f"     🌟 Best val MAE(real): {val_mae:.6f} | RMSE(real): {val_rmse:.6f} | "
                f"MAPE(real): {val_mape:.2f}% | WAPE(real): {val_wape:.2f}%"
            )

        if early_stopping.should_stop:
            print(f"\n  ⏹️ Early Stopping at epoch {epoch}")
            break

    train_time = time.time() - exp_start_time

    # ------ 6. 加载最佳模型进行评估 ------
    print(f"\n  📊 评估最佳模型...")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', -1)
        best_val_loss = checkpoint.get('val_loss', -1)
        best_val_rmse = checkpoint.get('val_rmse', -1)
        best_val_mape = checkpoint.get('val_mape', -1)
        best_val_wape = checkpoint.get('val_wape', -1)
    else:
        best_epoch = -1
        best_val_loss = -1
        best_val_rmse = -1
        best_val_mape = -1
        best_val_wape = -1

    eval_results = evaluate_model(model, test_loader, A_dist, A_sem, scaler, device, args)

    # ------ 7. 保存结果 ------
    results = {
        'experiment_name': exp_name,
        'description': getattr(args, '_description', ''),
        'config': {
            'npz_path': args.npz_path,
            'use_spatial_pe': args.use_spatial_pe,
            'use_pfga': args.use_pfga,
            'use_temporal_conv': args.use_temporal_conv,
            'use_dual_graph': args.use_dual_graph,
            'loss_alpha': args.loss_alpha,
            'loss_beta': args.loss_beta,
            'loss_gamma1': args.loss_gamma1,
            'loss_gamma2': args.loss_gamma2,
            'loss_delta': args.loss_delta,
            'loss_epsilon': args.loss_epsilon,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
        },
        'training': {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'best_val_rmse': best_val_rmse,
            'best_val_mape': best_val_mape,
            'best_val_wape': best_val_wape,
            'total_train_time_sec': round(train_time, 1),
            'num_params': num_params,
            'num_trainable_params': num_trainable,
        },
        'metrics': eval_results,
    }

    results_path = os.path.join(result_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ 实验 [{exp_name}] 完成！")
    print(f"     MAE:  {eval_results['overall']['mae']:.4f}")
    print(f"     RMSE: {eval_results['overall']['rmse']:.4f}")
    print(f"     MAPE: {eval_results['overall']['mape']:.2f}%")
    print(f"     WAPE: {eval_results['overall']['wape']:.2f}%")
    print(f"     训练时间: {train_time/60:.1f} 分钟")
    print(f"     结果已保存: {results_path}")

    # 清理 GPU 内存
    del model, optimizer, scaler_amp, A_dist, A_sem
    torch.cuda.empty_cache()

    return results


# ============================================================
# 结果汇总
# ============================================================

def generate_summary(results_base_dir, experiments_config):
    """生成所有实验的汇总对比表"""
    summary_lines = []
    summary_lines.append("# ST-LLM+ 消融实验结果汇总\n")
    summary_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 表头
    summary_lines.append("| 实验名 | 说明 | MAE | RMSE | MAPE(%) | WAPE(%) | 训练时间(min) | Best Epoch |")
    summary_lines.append("|--------|------|-----|------|---------|---------|---------------|------------|")

    csv_lines = ["experiment,description,MAE,RMSE,MAPE,WAPE,train_time_min,best_epoch"]

    for exp_name, exp_config in experiments_config.items():
        result_dir = os.path.join(results_base_dir, exp_name)
        results_path = os.path.join(result_dir, 'results.json')

        if not os.path.exists(results_path):
            summary_lines.append(f"| {exp_name} | {exp_config.get('description', '')} | - | - | - | - | - | - |")
            continue

        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        m = results['metrics']['overall']
        t = results['training']
        desc = exp_config.get('description', '')

        summary_lines.append(
            f"| {exp_name} | {desc} | "
            f"{m['mae']:.4f} | {m['rmse']:.4f} | "
            f"{m['mape']:.2f} | {m['wape']:.2f} | "
            f"{t['total_train_time_sec']/60:.1f} | {t['best_epoch']} |"
        )

        csv_lines.append(
            f"{exp_name},{desc},{m['mae']:.4f},{m['rmse']:.4f},"
            f"{m['mape']:.2f},{m['wape']:.2f},"
            f"{t['total_train_time_sec']/60:.1f},{t['best_epoch']}"
        )

    # 分步长指标表
    summary_lines.append("\n## 分步长指标 (MAE)\n")
    horizons_header = "| 实验名 |"
    horizons_sep = "|--------|"

    # 收集所有 horizon 名称
    all_horizons = set()
    for exp_name in experiments_config:
        rp = os.path.join(results_base_dir, exp_name, 'results.json')
        if os.path.exists(rp):
            with open(rp, 'r') as f:
                r = json.load(f)
            all_horizons.update(r['metrics'].get('per_horizon', {}).keys())

    sorted_horizons = sorted(all_horizons)
    for h in sorted_horizons:
        horizons_header += f" {h} |"
        horizons_sep += "------|"

    summary_lines.append(horizons_header)
    summary_lines.append(horizons_sep)

    for exp_name in experiments_config:
        rp = os.path.join(results_base_dir, exp_name, 'results.json')
        if not os.path.exists(rp):
            continue
        with open(rp, 'r') as f:
            r = json.load(f)
        ph = r['metrics'].get('per_horizon', {})
        row = f"| {exp_name} |"
        for h in sorted_horizons:
            if h in ph:
                row += f" {ph[h]['mae']:.4f} |"
            else:
                row += " - |"
        summary_lines.append(row)

    # 写入文件
    summary_path = os.path.join(results_base_dir, "summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    csv_path = os.path.join(results_base_dir, "summary.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_lines))

    print(f"\n📊 汇总表已生成:")
    print(f"   Markdown: {summary_path}")
    print(f"   CSV:      {csv_path}")

    # 打印到控制台
    print("\n" + "=" * 100)
    for line in summary_lines:
        print(line)
    print("=" * 100)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ST-LLM+ Ablation Study Runner")
    parser.add_argument("--config", type=str,
                        default="/kaggle/input/datasets/chengyy03/coding-v3/ST-LLM-Plus/configs/ablation_config.yaml",
                        help="消融实验配置文件路径")
    parser.add_argument("--batch", type=int, default=None,
                        help="只运行指定 batch 编号的实验")
    parser.add_argument("--only", type=str, default=None,
                        help="只运行指定名称的实验")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="跳过已有结果的实验")
    parser.add_argument("--summary-only", action="store_true", default=False,
                        help="仅生成汇总表（不训练）")

    cli_args = parser.parse_args()

    # 加载配置
    print(f"📄 加载消融配置: {cli_args.config}")
    config = load_ablation_config(cli_args.config)
    defaults = config.get('defaults', {})
    experiments = config.get('experiments', {})

    results_base_dir = defaults.get('results_base_dir', './ablation_results')
    os.makedirs(results_base_dir, exist_ok=True)

    # 仅汇总模式
    if cli_args.summary_only:
        generate_summary(results_base_dir, experiments)
        return

    # 获取基础 args
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        base_args = get_args()
    finally:
        sys.argv = original_argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 使用设备: {device}")

    # 筛选要运行的实验
    experiments_to_run = {}
    for exp_name, exp_config in experiments.items():
        if not exp_config.get('enabled', True):
            continue
        if cli_args.only and exp_name != cli_args.only:
            continue
        if cli_args.batch is not None and exp_config.get('batch') != cli_args.batch:
            continue
        experiments_to_run[exp_name] = exp_config

    total_exps = len(experiments_to_run)
    print(f"\n📋 共有 {total_exps} 个实验待运行:")
    for i, (name, cfg) in enumerate(experiments_to_run.items(), 1):
        status = "⏩ 已完成" if is_experiment_completed(
            os.path.join(results_base_dir, name)
        ) else "⏳ 待运行"
        print(f"   {i}. [{name}] {cfg.get('description', '')} — {status}")

    # 运行实验
    completed = 0
    skipped = 0
    failed = 0

    global_start = time.time()

    for i, (exp_name, exp_config) in enumerate(experiments_to_run.items(), 1):
        result_dir = os.path.join(results_base_dir, exp_name)

        # 断点续跑
        if cli_args.skip_completed and is_experiment_completed(result_dir):
            print(f"\n⏩ [{i}/{total_exps}] 跳过已完成的实验: {exp_name}")
            skipped += 1
            continue

        print(f"\n{'='*80}")
        print(f"🧪 [{i}/{total_exps}] 开始实验: {exp_name}")
        print(f"   说明: {exp_config.get('description', '')}")
        print(f"{'='*80}")

        # 合并参数
        args = merge_args(base_args, defaults, exp_config)
        args.experiment_name = exp_name
        args.save_dir = result_dir
        args._description = exp_config.get('description', '')

        try:
            run_single_experiment(exp_name, args, result_dir, device)
            completed += 1
        except Exception as e:
            print(f"\n  ❌ 实验 [{exp_name}] 失败: {e}")
            import traceback
            traceback.print_exc()

            # 保存错误信息
            error_path = os.path.join(result_dir, 'error.json')
            os.makedirs(result_dir, exist_ok=True)
            with open(error_path, 'w') as f:
                json.dump({'error': str(e), 'traceback': traceback.format_exc()}, f)

            failed += 1
            continue

        # 时间估算
        elapsed = time.time() - global_start
        remaining = total_exps - i
        if completed > 0:
            avg_time = elapsed / completed
            est_remaining = avg_time * remaining
            print(f"\n  ⏱️ 已用时: {elapsed/60:.1f}min | "
                  f"预计剩余: {est_remaining/60:.1f}min")

    # 生成汇总
    total_time = time.time() - global_start
    print(f"\n{'='*80}")
    print(f"🏁 消融实验批量运行完成！")
    print(f"   完成: {completed} | 跳过: {skipped} | 失败: {failed}")
    print(f"   总耗时: {total_time/60:.1f} 分钟")
    print(f"{'='*80}")

    generate_summary(results_base_dir, experiments)


if __name__ == "__main__":
    main()
