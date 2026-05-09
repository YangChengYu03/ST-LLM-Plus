"""
ST-LLM+ evaluation entry.

Features:
1. Load the best checkpoint.
2. Run inference on the test split.
3. Inverse-transform predictions.
4. Report overall and per-horizon metrics.
5. Save plots and metric files for Kaggle outputs.
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

from configs.default import get_args
from data.dataset import create_dataloaders
from models.st_llm import STLLMModel
from utils.metrics import compute_metrics, compute_per_horizon_metrics


def load_model(args, device):
    """Load a trained checkpoint if it exists."""
    model = STLLMModel(args).to(device)

    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model: {args.model_path}")
        if "epoch" in checkpoint:
            print(f"  checkpoint epoch: {checkpoint['epoch']}")
        if "val_loss" in checkpoint:
            print(f"  checkpoint val_loss: {checkpoint['val_loss']:.6f}")
    else:
        print(f"Warning: checkpoint not found: {args.model_path}")
        print("  Evaluation will run with randomly initialized weights.")

    return model


@torch.no_grad()
def inference(model, test_loader, A_dist, A_sem, device, use_amp=True):
    """Run inference on the test set."""
    model.eval()
    all_preds = []
    all_trues = []

    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.float16, enabled=use_amp):
            preds = model(batch_x, A_dist, A_sem)

        all_preds.append(preds.float().cpu().numpy())
        all_trues.append(batch_y.float().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze(-1)
    all_trues = np.concatenate(all_trues, axis=0).squeeze(-1)
    return all_preds, all_trues


def inverse_transform_results(preds, trues, scaler, feature_idx=0):
    """Inverse-transform predictions and labels."""
    preds_inv = scaler.inverse_transform(preds, feature_idx=feature_idx)
    trues_inv = scaler.inverse_transform(trues, feature_idx=feature_idx)
    return preds_inv, trues_inv


def save_metrics_report(overall, per_horizon, save_dir):
    """Save metrics as JSON and CSV."""
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, "metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "per_horizon": per_horizon}, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(save_dir, "metrics_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "mae", "rmse", "mape", "wape"])
        writer.writerow(["overall", overall["mae"], overall["rmse"], overall["mape"], overall["wape"]])
        for horizon_name, metrics in per_horizon.items():
            writer.writerow([horizon_name, metrics["mae"], metrics["rmse"], metrics["mape"], metrics["wape"]])


def visualize_results(preds_inv, trues_inv, args, save_dir="./results"):
    """
    Save time-series and scatter plots for selected nodes and horizons.
    """
    os.makedirs(save_dir, exist_ok=True)
    plot_length = min(args.plot_length, preds_inv.shape[0])

    candidate_steps = [0, min(5, args.output_len - 1), args.output_len - 1]
    candidate_steps = sorted(set(step for step in candidate_steps if 0 <= step < preds_inv.shape[1]))

    for node_id in args.plot_nodes:
        if node_id >= preds_inv.shape[2]:
            print(f"Skip node {node_id}: out of range")
            continue

        for step in candidate_steps:
            y_true = trues_inv[:plot_length, step, node_id]
            y_pred = preds_inv[:plot_length, step, node_id]

            fig, axes = plt.subplots(1, 2, figsize=(18, 5))

            axes[0].plot(y_true, label="Ground Truth", color="#1f77b4", linewidth=2.0, alpha=0.85)
            axes[0].plot(y_pred, label="Prediction", color="#d62728", linestyle="--", linewidth=2.0, alpha=0.85)
            axes[0].set_title(f"Node {node_id} | Step {step + 1}")
            axes[0].set_xlabel("Sample Index")
            axes[0].set_ylabel("Target Value")
            axes[0].legend()
            axes[0].grid(True, linestyle="--", alpha=0.35)

            y_true_all = trues_inv[:, step, node_id]
            y_pred_all = preds_inv[:, step, node_id]
            axes[1].scatter(y_true_all, y_pred_all, color="#2ca02c", alpha=0.35, s=12, edgecolors="none")

            min_val = min(float(y_true_all.min()), float(y_pred_all.min()))
            max_val = max(float(y_true_all.max()), float(y_pred_all.max()))
            axes[1].plot([min_val, max_val], [min_val, max_val], color="#333333", linestyle="--", linewidth=2)
            axes[1].set_title(f"Scatter | Node {node_id} Step {step + 1}")
            axes[1].set_xlabel("True")
            axes[1].set_ylabel("Predicted")
            axes[1].grid(True, linestyle="--", alpha=0.35)

            plt.tight_layout()
            fig_path = os.path.join(save_dir, f"node{node_id}_step{step + 1}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {fig_path}")


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_amp = bool(args.use_amp and device.type == "cuda")
    print(f"Using device: {device}")

    if not args.model_path:
        args.model_path = os.path.join(args.save_dir, "best_model.pth")

    _, _, test_loader, scaler = create_dataloaders(args)
    A_dist = torch.FloatTensor(np.load(args.adj_dist_path)).to(device)
    A_sem = torch.FloatTensor(np.load(args.adj_sem_path)).to(device)

    model = load_model(args, device)

    print("\nRunning inference on test split...")
    preds, trues = inference(model, test_loader, A_dist, A_sem, device, args.use_amp)
    print(f"Prediction shape: {preds.shape}, Target shape: {trues.shape}")

    preds_inv, trues_inv = inverse_transform_results(preds, trues, scaler)

    overall = compute_metrics(preds_inv, trues_inv, mape_threshold=args.mape_threshold)
    per_horizon = compute_per_horizon_metrics(
        preds_inv,
        trues_inv,
        horizons=args.eval_horizons,
        mape_threshold=args.mape_threshold,
    )

    print("\nOverall metrics (inverse transformed):")
    print(f"  MAE:  {overall['mae']:.4f}")
    print(f"  RMSE: {overall['rmse']:.4f}")
    print(f"  MAPE: {overall['mape']:.2f}%")
    print(f"  WAPE: {overall['wape']:.2f}%")

    print("\nPer-horizon metrics:")
    for horizon_name, metrics in per_horizon.items():
        print(
            f"  {horizon_name}: "
            f"MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"MAPE={metrics['mape']:.2f}% | "
            f"WAPE={metrics['wape']:.2f}%"
        )

    result_dir = os.path.join(args.save_dir, "eval_results")
    save_metrics_report(overall, per_horizon, result_dir)
    visualize_results(preds_inv, trues_inv, args, save_dir=result_dir)

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()
