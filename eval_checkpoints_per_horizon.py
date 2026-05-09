"""
Re-evaluate ablation checkpoints on the test split and export per-horizon metrics.

Use this when results.json is missing or corrupted. The script does not read
results.json. It loads each experiment's best_model.pth, rebuilds the matching
model from the YAML config, runs test-set inference, and writes MAE/RMSE/MAPE/WAPE
for every configured prediction horizon.

Examples:
    python eval_checkpoints_per_horizon.py \
      --config configs/ablation_config.yaml \
      --results_dir ./ablation_results \
      --output_dir ./recomputed_test_metrics

    python eval_checkpoints_per_horizon.py \
      --config configs/ablation_config.yaml \
      --results_dir ./ablation_results \
      --batch 2 \
      --output_dir ./recomputed_feature_metrics
"""

import argparse
import copy
import csv
import json
import os
import sys
import tempfile

import numpy as np
import torch
import yaml
from torch.amp import autocast

from configs.default import get_args
from data.dataset import create_dataloaders
from data.graph_builder import build_dist_graph, build_semantic_graph
from models.st_llm import STLLMModel
from utils.metrics import compute_metrics, compute_per_horizon_metrics


def load_base_args():
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return get_args()
    finally:
        sys.argv = original_argv


def merge_args(base_args, defaults, experiment_config):
    args = copy.deepcopy(base_args)

    for key, value in defaults.items():
        if key not in ("npz_base_dir", "results_base_dir", "coords_csv_path"):
            if hasattr(args, key):
                setattr(args, key, value)

    for key, value in experiment_config.items():
        if key in ("enabled", "batch", "npz_file", "description"):
            continue
        if hasattr(args, key):
            setattr(args, key, value)

    args.npz_path = os.path.join(defaults.get("npz_base_dir", ""), experiment_config.get("npz_file", "full.npz"))
    if "coords_csv_path" in defaults:
        args.coords_csv_path = defaults["coords_csv_path"]
    return args


def ensure_graphs(args, exp_result_dir, graph_cache_dir):
    source_adj_dir = os.path.join(exp_result_dir, "adj_data")
    source_dist = os.path.join(source_adj_dir, "adj_distance.npy")
    source_sem = os.path.join(source_adj_dir, "adj_semantic.npy")

    if os.path.exists(source_dist) and os.path.exists(source_sem):
        args.adj_dist_path = source_dist
        args.adj_sem_path = source_sem
        return

    os.makedirs(graph_cache_dir, exist_ok=True)
    args.adj_dist_path = os.path.join(graph_cache_dir, "adj_distance.npy")
    args.adj_sem_path = os.path.join(graph_cache_dir, "adj_semantic.npy")

    if not os.path.exists(args.adj_dist_path):
        build_dist_graph(
            npz_path=args.npz_path,
            coords_csv_path=args.coords_csv_path,
            save_path=args.adj_dist_path,
            sigma=args.dist_sigma,
            epsilon=args.dist_epsilon,
        )
    if not os.path.exists(args.adj_sem_path):
        build_semantic_graph(
            npz_path=args.npz_path,
            save_path=args.adj_sem_path,
            poi_features=args.poi_features,
            ntl_feature=args.ntl_feature,
            threshold=args.sem_threshold,
        )


@torch.no_grad()
def evaluate_checkpoint(model, test_loader, A_dist, A_sem, scaler, device, args):
    model.eval()
    all_preds = []
    all_trues = []

    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.float16, enabled=args.use_amp):
            preds = model(batch_x, A_dist, A_sem)
        all_preds.append(preds.float().cpu().numpy())
        all_trues.append(batch_y.float().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).squeeze(-1)
    trues = np.concatenate(all_trues, axis=0).squeeze(-1)
    preds_inv = scaler.inverse_transform(preds, feature_idx=0)
    trues_inv = scaler.inverse_transform(trues, feature_idx=0)

    return {
        "overall": compute_metrics(preds_inv, trues_inv, mape_threshold=args.mape_threshold),
        "per_horizon": compute_per_horizon_metrics(
            preds_inv,
            trues_inv,
            horizons=args.eval_horizons,
            mape_threshold=args.mape_threshold,
        ),
    }


def horizon_sort_key(name):
    try:
        return int(str(name).split("_")[-1])
    except ValueError:
        return 10**9


def write_outputs(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    long_path = os.path.join(output_dir, "test_per_horizon_metrics.csv")
    with open(long_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "description", "horizon", "MAE", "RMSE", "MAPE", "WAPE", "checkpoint"],
        )
        writer.writeheader()
        writer.writerows(rows)

    experiments = {}
    horizons = set()
    for row in rows:
        experiments.setdefault(row["experiment"], {"description": row["description"], "checkpoint": row["checkpoint"], "metrics": {}})
        experiments[row["experiment"]]["metrics"][row["horizon"]] = row
        horizons.add(row["horizon"])

    sorted_horizons = sorted(horizons, key=horizon_sort_key)
    wide_path = os.path.join(output_dir, "test_per_horizon_metrics_wide.csv")
    header = ["experiment", "description"]
    for horizon in sorted_horizons:
        header.extend([f"{horizon}_MAE", f"{horizon}_RMSE", f"{horizon}_MAPE", f"{horizon}_WAPE"])
    header.append("checkpoint")

    with open(wide_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for exp_name, payload in experiments.items():
            line = [exp_name, payload["description"]]
            for horizon in sorted_horizons:
                metric = payload["metrics"].get(horizon)
                if metric:
                    line.extend([metric["MAE"], metric["RMSE"], metric["MAPE"], metric["WAPE"]])
                else:
                    line.extend(["", "", "", ""])
            line.append(payload["checkpoint"])
            writer.writerow(line)

    json_path = os.path.join(output_dir, "test_per_horizon_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nWrote:")
    print(f"  {long_path}")
    print(f"  {wide_path}")
    print(f"  {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and export test per-horizon metrics")
    parser.add_argument("--config", type=str, required=True, help="Experiment YAML config")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing experiment checkpoints")
    parser.add_argument("--output_dir", type=str, default="./recomputed_test_metrics")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="", help="Optional experiment-name prefix filter, e.g. feat_ or mod_")
    args_cli = parser.parse_args()

    with open(args_cli.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    experiments = config.get("experiments", {})
    base_args = load_base_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = []
    temp_root = tempfile.mkdtemp(prefix="stllm_eval_graphs_")

    for exp_name, exp_config in experiments.items():
        if not exp_config.get("enabled", True):
            continue
        if args_cli.only and exp_name != args_cli.only:
            continue
        if args_cli.batch is not None and exp_config.get("batch") != args_cli.batch:
            continue
        if args_cli.prefix and not exp_name.startswith(args_cli.prefix):
            continue

        exp_result_dir = os.path.join(args_cli.results_dir, exp_name)
        checkpoint_path = os.path.join(exp_result_dir, "best_model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Skip {exp_name}: missing checkpoint {checkpoint_path}")
            continue

        print(f"\nEvaluating {exp_name}")
        exp_args = merge_args(base_args, defaults, exp_config)
        exp_args.experiment_name = exp_name
        exp_args.save_dir = exp_result_dir
        exp_args.use_amp = bool(exp_args.use_amp and device.type == "cuda")
        graph_cache_dir = os.path.join(temp_root, exp_name)
        ensure_graphs(exp_args, exp_result_dir, graph_cache_dir)

        _, _, test_loader, scaler = create_dataloaders(exp_args)
        A_dist = torch.FloatTensor(np.load(exp_args.adj_dist_path)).to(device)
        A_sem = torch.FloatTensor(np.load(exp_args.adj_sem_path)).to(device)

        model = STLLMModel(exp_args).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        metrics = evaluate_checkpoint(model, test_loader, A_dist, A_sem, scaler, device, exp_args)
        description = exp_config.get("description", "")

        for horizon, horizon_metrics in sorted(metrics["per_horizon"].items(), key=lambda item: horizon_sort_key(item[0])):
            rows.append(
                {
                    "experiment": exp_name,
                    "description": description,
                    "horizon": horizon,
                    "MAE": f"{horizon_metrics['mae']:.6f}",
                    "RMSE": f"{horizon_metrics['rmse']:.6f}",
                    "MAPE": f"{horizon_metrics['mape']:.4f}",
                    "WAPE": f"{horizon_metrics['wape']:.4f}",
                    "checkpoint": checkpoint_path,
                }
            )

        overall = metrics["overall"]
        print(
            f"  overall: MAE={overall['mae']:.4f}, RMSE={overall['rmse']:.4f}, "
            f"MAPE={overall['mape']:.2f}%, WAPE={overall['wape']:.2f}%"
        )

        del model, A_dist, A_sem
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        print("No checkpoint metrics were generated.")
        return

    write_outputs(rows, args_cli.output_dir)


if __name__ == "__main__":
    main()
