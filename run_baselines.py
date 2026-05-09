"""
Batch runner for classic traffic forecasting baselines.

Usage:
    python run_baselines.py --config configs/baseline_config.yaml
    python run_baselines.py --config configs/baseline_config.yaml --only agcrn
    python run_baselines.py --config configs/baseline_config.yaml --summary-only
"""

import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast

from configs.default import get_args
from data.dataset import create_dataloaders
from data.graph_builder import build_dist_graph, build_semantic_graph
from models.baselines import build_baseline_model
from models.losses import CombinedTrafficLoss
from utils.logger import EarlyStopping, TrainLogger
from utils.metrics import compute_metrics, compute_per_horizon_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_real_metrics(preds, targets, scaler, mape_threshold=0.01):
    preds_real = scaler.inverse_transform(preds.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
    targets_real = scaler.inverse_transform(targets.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
    metrics = compute_metrics(preds_real, targets_real, mape_threshold=mape_threshold)
    return {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "mape": float(metrics["mape"]),
        "wape": float(metrics["wape"]),
    }


def train_one_epoch(model, train_loader, criterion, optimizer, scaler_amp,
                    A_dist, A_sem, device, max_grad_norm, use_amp, scaler, mape_threshold=0.01):
    model.train()
    total_loss = 0.0
    total_metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "wape": 0.0}
    loss_components = {"huber": 0.0, "quantile": 0.0, "tc": 0.0, "cp": 0.0, "vp": 0.0, "rc": 0.0}
    num_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast("cuda", dtype=torch.float16, enabled=use_amp):
            preds = model(batch_x, A_dist, A_sem)
            loss, loss_dict = criterion(preds, batch_y)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss_dict["total"]
        for key in loss_components:
            loss_components[key] += loss_dict.get(key, 0.0)
        batch_metrics = compute_real_metrics(preds, batch_y, scaler, mape_threshold=mape_threshold)
        for key in total_metrics:
            total_metrics[key] += batch_metrics[key]
        num_batches += 1

    denom = max(num_batches, 1)
    avg_metrics = {key: value / denom for key, value in total_metrics.items()}
    avg_components = {key: value / denom for key, value in loss_components.items()}
    return total_loss / denom, avg_metrics, avg_components


@torch.no_grad()
def validate(model, val_loader, criterion, A_dist, A_sem, device, use_amp, scaler, mape_threshold=0.01):
    model.eval()
    total_loss = 0.0
    preds_real_all = []
    trues_real_all = []
    num_batches = 0

    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.float16, enabled=use_amp):
            preds = model(batch_x, A_dist, A_sem)
            _, loss_dict = criterion(preds, batch_y)

        total_loss += loss_dict["total"]
        preds_real_all.append(
            scaler.inverse_transform(preds.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
        )
        trues_real_all.append(
            scaler.inverse_transform(batch_y.detach().float().cpu().numpy().squeeze(-1), feature_idx=0)
        )
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    if preds_real_all:
        preds_real = np.concatenate(preds_real_all, axis=0)
        trues_real = np.concatenate(trues_real_all, axis=0)
        metrics = compute_metrics(preds_real, trues_real, mape_threshold=mape_threshold)
    else:
        metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "wape": 0.0}
    return avg_loss, metrics


@torch.no_grad()
def evaluate_model(model, test_loader, A_dist, A_sem, scaler, device, args):
    model.eval()
    all_preds = []
    all_trues = []

    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.float16, enabled=args.use_amp):
            preds = model(batch_x, A_dist, A_sem)
        all_preds.append(preds.float().cpu().numpy())
        all_trues.append(batch_y.float().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze(-1)
    all_trues = np.concatenate(all_trues, axis=0).squeeze(-1)
    preds_inv = scaler.inverse_transform(all_preds, feature_idx=0)
    trues_inv = scaler.inverse_transform(all_trues, feature_idx=0)

    return {
        "overall": compute_metrics(preds_inv, trues_inv, mape_threshold=args.mape_threshold),
        "per_horizon": compute_per_horizon_metrics(
            preds_inv, trues_inv, horizons=args.eval_horizons, mape_threshold=args.mape_threshold
        ),
    }


def generate_summary(results_base_dir, experiments_config):
    summary_lines = ["# Classic Baseline Results\n", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    summary_lines.append("| Experiment | Description | MAE | RMSE | MAPE(%) | WAPE(%) | Train Time(min) | Best Epoch |")
    summary_lines.append("|--------|------|-----|------|---------|---------|---------------|------------|")
    csv_lines = ["experiment,description,MAE,RMSE,MAPE,WAPE,train_time_min,best_epoch"]

    for exp_name, exp_config in experiments_config.items():
        results_path = os.path.join(results_base_dir, exp_name, "results.json")
        if not os.path.exists(results_path):
            summary_lines.append(f"| {exp_name} | {exp_config.get('description', '')} | - | - | - | - | - | - |")
            continue
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        m = results["metrics"]["overall"]
        t = results["training"]
        desc = exp_config.get("description", "")
        summary_lines.append(
            f"| {exp_name} | {desc} | {m['mae']:.4f} | {m['rmse']:.4f} | "
            f"{m['mape']:.2f} | {m['wape']:.2f} | {t['total_train_time_sec']/60:.1f} | {t['best_epoch']} |"
        )
        csv_lines.append(
            f"{exp_name},{desc},{m['mae']:.4f},{m['rmse']:.4f},{m['mape']:.2f},"
            f"{m['wape']:.2f},{t['total_train_time_sec']/60:.1f},{t['best_epoch']}"
        )

    summary_lines.append("\n## Per-Horizon MAE\n")
    all_horizons = set()
    for exp_name in experiments_config:
        results_path = os.path.join(results_base_dir, exp_name, "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                all_horizons.update(json.load(f)["metrics"].get("per_horizon", {}).keys())

    sorted_horizons = sorted(all_horizons)
    summary_lines.append("| Experiment |" + "".join(f" {h} |" for h in sorted_horizons))
    summary_lines.append("|--------|" + "".join("------|" for _ in sorted_horizons))
    for exp_name in experiments_config:
        results_path = os.path.join(results_base_dir, exp_name, "results.json")
        if not os.path.exists(results_path):
            continue
        with open(results_path, "r", encoding="utf-8") as f:
            per_horizon = json.load(f)["metrics"].get("per_horizon", {})
        row = f"| {exp_name} |"
        for horizon in sorted_horizons:
            row += f" {per_horizon[horizon]['mae']:.4f} |" if horizon in per_horizon else " - |"
        summary_lines.append(row)

    with open(os.path.join(results_base_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    with open(os.path.join(results_base_dir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
    print(f"Summary written to {results_base_dir}")


def merge_args(base_args, defaults, experiment_overrides):
    args = copy.deepcopy(base_args)

    for key, value in defaults.items():
        if key not in ("npz_base_dir", "results_base_dir", "coords_csv_path"):
            setattr(args, key, value)

    for key, value in experiment_overrides.items():
        if key in ("enabled", "batch", "npz_file", "description", "model"):
            continue
        setattr(args, key, value)

    args.npz_path = os.path.join(defaults.get("npz_base_dir", ""), experiment_overrides.get("npz_file", "full.npz"))
    if "coords_csv_path" in defaults:
        args.coords_csv_path = defaults["coords_csv_path"]
    return args


def is_experiment_completed(result_dir):
    return os.path.exists(os.path.join(result_dir, "results.json"))


def build_graphs(args, result_dir, device):
    adj_dir = os.path.join(result_dir, "adj_data")
    os.makedirs(adj_dir, exist_ok=True)
    adj_dist_path = os.path.join(adj_dir, "adj_distance.npy")
    adj_sem_path = os.path.join(adj_dir, "adj_semantic.npy")

    if not os.path.exists(adj_dist_path):
        build_dist_graph(
            npz_path=args.npz_path,
            coords_csv_path=args.coords_csv_path,
            save_path=adj_dist_path,
            sigma=args.dist_sigma,
            epsilon=args.dist_epsilon,
        )

    if not os.path.exists(adj_sem_path):
        build_semantic_graph(
            npz_path=args.npz_path,
            save_path=adj_sem_path,
            poi_features=args.poi_features,
            ntl_feature=args.ntl_feature,
            threshold=args.sem_threshold,
        )

    return (
        torch.FloatTensor(np.load(adj_dist_path)).to(device),
        torch.FloatTensor(np.load(adj_sem_path)).to(device),
    )


def run_single_baseline(exp_name, exp_config, args, result_dir, device):
    os.makedirs(result_dir, exist_ok=True)
    set_seed(args.seed)
    start_time = time.time()

    A_dist, A_sem = build_graphs(args, result_dir, device)
    train_loader, val_loader, test_loader, scaler = create_dataloaders(args)

    model_name = exp_config.get("model", exp_name)
    print(f"\nBuilding baseline model: {model_name}")
    model = build_baseline_model(model_name, args).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params: {num_params:,}")
    print(f"  trainable params: {num_trainable:,}")

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
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler_amp = GradScaler("cuda", enabled=args.use_amp)
    logger = TrainLogger(log_dir=result_dir)
    early_stopping = EarlyStopping(patience=args.early_stop_patience)
    best_model_path = os.path.join(result_dir, "best_model.pth")

    print(f"\nStart training {exp_name} for up to {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_metrics, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler_amp,
            A_dist, A_sem, device, args.max_grad_norm, args.use_amp, scaler,
            mape_threshold=args.mape_threshold,
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, A_dist, A_sem, device, args.use_amp, scaler,
            mape_threshold=args.mape_threshold,
        )
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:03d}/{args.epochs} | Time: {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train MAE(real): {train_metrics['mae']:.4f} | "
            f"Val MAE(real): {val_metrics['mae']:.4f} | "
            f"Val RMSE(real): {val_metrics['rmse']:.4f} | "
            f"Val MAPE(real): {val_metrics['mape']:.2f}% | "
            f"Val WAPE(real): {val_metrics['wape']:.2f}%"
        )

        logger.log_epoch(
            epoch,
            train_loss,
            val_loss,
            optimizer.param_groups[0]["lr"],
            epoch_time,
            extra_metrics={
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_mape": train_metrics["mape"],
                "train_wape": train_metrics["wape"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_mape": val_metrics["mape"],
                "val_wape": val_metrics["wape"],
            },
        )

        if early_stopping(val_metrics["mae"]):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mape": val_metrics["mape"],
                    "val_wape": val_metrics["wape"],
                    "experiment_name": exp_name,
                    "model_name": model_name,
                },
                best_model_path,
            )

        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch = checkpoint.get("epoch", -1)
        best_val_loss = checkpoint.get("val_loss", -1)
        best_val_rmse = checkpoint.get("val_rmse", -1)
        best_val_mape = checkpoint.get("val_mape", -1)
        best_val_wape = checkpoint.get("val_wape", -1)
    else:
        best_epoch = -1
        best_val_loss = -1
        best_val_rmse = -1
        best_val_mape = -1
        best_val_wape = -1

    eval_results = evaluate_model(model, test_loader, A_dist, A_sem, scaler, device, args)
    results = {
        "experiment_name": exp_name,
        "description": exp_config.get("description", ""),
        "config": {
            "model": model_name,
            "npz_path": args.npz_path,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "baseline_hidden_dim": getattr(args, "baseline_hidden_dim", None),
        },
        "training": {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_rmse": best_val_rmse,
            "best_val_mape": best_val_mape,
            "best_val_wape": best_val_wape,
            "total_train_time_sec": round(train_time, 1),
            "num_params": num_params,
            "num_trainable_params": num_trainable,
        },
        "metrics": eval_results,
    }

    results_path = os.path.join(result_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted {exp_name}")
    print(f"  MAE:  {eval_results['overall']['mae']:.4f}")
    print(f"  RMSE: {eval_results['overall']['rmse']:.4f}")
    print(f"  MAPE: {eval_results['overall']['mape']:.2f}%")
    print(f"  WAPE: {eval_results['overall']['wape']:.2f}%")
    print(f"  results: {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Classic baseline comparison runner")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--summary-only", action="store_true", default=False)
    cli_args = parser.parse_args()

    with open(cli_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    experiments = config.get("experiments", {})
    results_base_dir = defaults.get("results_base_dir", "./baseline_results")
    os.makedirs(results_base_dir, exist_ok=True)

    if cli_args.summary_only:
        generate_summary(results_base_dir, experiments)
        return

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        base_args = get_args()
    finally:
        sys.argv = original_argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    experiments_to_run = {}
    for exp_name, exp_config in experiments.items():
        if not exp_config.get("enabled", True):
            continue
        if cli_args.only and exp_name != cli_args.only:
            continue
        if cli_args.batch is not None and exp_config.get("batch") != cli_args.batch:
            continue
        experiments_to_run[exp_name] = exp_config

    print(f"Found {len(experiments_to_run)} baseline experiments")
    completed = skipped = failed = 0
    for exp_name, exp_config in experiments_to_run.items():
        result_dir = os.path.join(results_base_dir, exp_name)
        if cli_args.skip_completed and is_experiment_completed(result_dir):
            print(f"Skip completed experiment: {exp_name}")
            skipped += 1
            continue

        args = merge_args(base_args, defaults, exp_config)
        args.experiment_name = exp_name
        args.save_dir = result_dir
        args.use_amp = bool(args.use_amp and device.type == "cuda")

        try:
            run_single_baseline(exp_name, exp_config, args, result_dir, device)
            completed += 1
        except Exception as exc:
            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, "error.json"), "w", encoding="utf-8") as f:
                json.dump({"error": str(exc)}, f, indent=2, ensure_ascii=False)
            print(f"Experiment failed: {exp_name}: {exc}")
            failed += 1

    print(f"\nBaseline run finished. completed={completed}, skipped={skipped}, failed={failed}")
    generate_summary(results_base_dir, experiments)


if __name__ == "__main__":
    main()
