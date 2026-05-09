"""
ST-LLM+ training entry.

Features:
1. Set random seed.
2. Build graph files when missing.
3. Load and standardize data.
4. Build model.
5. Train with Ranger and optional AMP.
6. Early stop on real-space validation MAE.
7. Save the best checkpoint and training curves.
"""

import os
import random
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from configs.default import get_args
from data.dataset import create_dataloaders
from data.graph_builder import build_dist_graph, build_semantic_graph
from models.losses import CombinedTrafficLoss
from models.st_llm import STLLMModel
from ranger21 import Ranger
from utils.logger import EarlyStopping, TrainLogger
from utils.metrics import compute_metrics


def set_seed(seed):
    """Make runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_real_metrics(preds, targets, scaler, mape_threshold=0.01):
    """Compute MAE/RMSE/MAPE/WAPE after inverse transform."""
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
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_real_mae = 0.0
    total_real_rmse = 0.0
    total_real_mape = 0.0
    total_real_wape = 0.0
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

        real_metrics = compute_real_metrics(preds, batch_y, scaler, mape_threshold=mape_threshold)
        total_real_mae += real_metrics["mae"]
        total_real_rmse += real_metrics["rmse"]
        total_real_mape += real_metrics["mape"]
        total_real_wape += real_metrics["wape"]
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    train_metrics = {
        "mae": total_real_mae / max(num_batches, 1),
        "rmse": total_real_rmse / max(num_batches, 1),
        "mape": total_real_mape / max(num_batches, 1),
        "wape": total_real_wape / max(num_batches, 1),
    }
    for key in loss_components:
        loss_components[key] /= max(num_batches, 1)

    return avg_loss, train_metrics, loss_components


@torch.no_grad()
def validate(model, val_loader, criterion, A_dist, A_sem, device, use_amp, scaler, mape_threshold=0.01):
    """Validate for one epoch and return normalized loss plus real-space metrics."""
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
        metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "wape": 0.0}

    return avg_loss, metrics


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_amp = bool(args.use_amp and device.type == "cuda")
    print(f"Using device: {device}")

    if not os.path.exists(args.adj_dist_path):
        print("Building distance graph...")
        build_dist_graph(
            npz_path=args.npz_path,
            coords_csv_path=args.coords_csv_path,
            save_path=args.adj_dist_path,
            sigma=args.dist_sigma,
            epsilon=args.dist_epsilon,
        )

    if not os.path.exists(args.adj_sem_path):
        print("Building semantic graph...")
        build_semantic_graph(
            npz_path=args.npz_path,
            save_path=args.adj_sem_path,
            poi_features=args.poi_features,
            ntl_feature=args.ntl_feature,
            threshold=args.sem_threshold,
        )

    A_dist = torch.FloatTensor(np.load(args.adj_dist_path)).to(device)
    A_sem = torch.FloatTensor(np.load(args.adj_sem_path)).to(device)
    print(f"Loaded graphs: A_dist{tuple(A_dist.shape)}, A_sem{tuple(A_sem.shape)}")

    train_loader, val_loader, test_loader, scaler = create_dataloaders(args)
    del test_loader

    print("\nBuilding ST-LLM+ model...")
    model = STLLMModel(args).to(device)
    model.ensure_float32_trainable()

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

    optimizer = Ranger(
        model.get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler_amp = GradScaler("cuda", enabled=args.use_amp)

    logger = TrainLogger(log_dir=args.save_dir)
    early_stopping = EarlyStopping(patience=args.early_stop_patience)
    best_model_path = os.path.join(args.save_dir, "best_model.pth")

    print(f"\nStart training for up to {args.epochs} epochs")
    print(f"  optimizer: Ranger | lr: {args.lr}")
    print(f"  early stopping metric: val MAE(real), patience={args.early_stop_patience}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_metrics, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler_amp,
            A_dist, A_sem, device, args.max_grad_norm, args.use_amp, scaler,
            mape_threshold=args.mape_threshold
        )
        train_mae = train_metrics["mae"]
        train_rmse = train_metrics["rmse"]
        train_mape = train_metrics["mape"]
        train_wape = train_metrics["wape"]
        val_loss, val_metrics = validate(
            model, val_loader, criterion, A_dist, A_sem, device, args.use_amp, scaler,
            mape_threshold=args.mape_threshold
        )
        val_mae = val_metrics["mae"]
        val_rmse = val_metrics["rmse"]
        val_mape = val_metrics["mape"]
        val_wape = val_metrics["wape"]

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
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
            f"LR: {current_lr:.6f}"
        )

        logger.log_epoch(
            epoch,
            train_loss,
            val_loss,
            current_lr,
            epoch_time,
            extra_metrics={
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "train_mape": train_mape,
                "train_wape": train_wape,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_mape": val_mape,
                "val_wape": val_wape,
            },
        )

        is_best = early_stopping(val_mae)
        if is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_mae": train_mae,
                    "train_rmse": train_rmse,
                    "train_mape": train_mape,
                    "train_wape": train_wape,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_mape": val_mape,
                    "val_wape": val_wape,
                    "args": args,
                },
                best_model_path,
            )
            print(
                f"  Saved new best checkpoint with val MAE(real)={val_mae:.6f} | "
                f"RMSE={val_rmse:.6f} | MAPE={val_mape:.2f}% | WAPE={val_wape:.2f}%"
            )

        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered after epoch {epoch}.")
            break

    print("=" * 80)
    print(f"Training finished. Best checkpoint: {best_model_path}")
    print(f"Best val MAE(real): {early_stopping.best_loss:.6f}")
    print(f"Loss curve saved to: {logger.plot_path}")


if __name__ == "__main__":
    main()
