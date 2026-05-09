"""
Training logger utilities.
"""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainLogger:
    """
    Save training logs to txt/csv/json and render loss curves.
    Supports extra per-epoch metrics via `extra_metrics`.
    """

    def __init__(self, log_dir="./logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"train_log_{timestamp}.txt")
        self.csv_path = os.path.join(log_dir, "train_history.csv")
        self.json_path = os.path.join(log_dir, "train_history.json")
        self.plot_path = os.path.join(log_dir, "loss_curve.png")
        self.history = []
        self.extra_fields = []
        self._write_header()

    def _write_header(self):
        extra_csv = "," + ",".join(self.extra_fields) if self.extra_fields else ""
        extra_txt = "\t" + "\t".join(self.extra_fields) if self.extra_fields else ""

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"epoch\ttrain_loss\tval_loss\tlr\ttime{extra_txt}\n")
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write(f"epoch,train_loss,val_loss,lr,epoch_time_sec{extra_csv}\n")

    def log_epoch(self, epoch, train_loss, val_loss, lr, epoch_time, extra_metrics=None):
        extra_metrics = extra_metrics or {}

        # Initialize extra metric columns on first use.
        if not self.extra_fields and extra_metrics:
            self.extra_fields = list(extra_metrics.keys())
            self._write_header()

        extra_txt = ""
        extra_csv_values = ""
        for key in self.extra_fields:
            value = extra_metrics.get(key, None)
            if value is None:
                extra_txt += "\t"
                extra_csv_values += ","
            else:
                extra_txt += f"\t{float(value):.6f}"
                extra_csv_values += f",{float(value):.6f}"

        line = f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\t{lr:.8f}\t{epoch_time:.1f}s{extra_txt}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{lr:.8f},{epoch_time:.4f}{extra_csv_values}\n")

        record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(lr),
            "epoch_time_sec": float(epoch_time),
        }
        for key in self.extra_fields:
            value = extra_metrics.get(key, None)
            record[key] = None if value is None else float(value)

        self.history.append(record)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        self._save_loss_curve()
        print(line)

    def log_message(self, msg):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"# {msg}\n")
        print(msg)

    def _save_loss_curve(self):
        if not self.history:
            return

        epochs = [item["epoch"] for item in self.history]
        train_loss = [item["train_loss"] for item in self.history]
        val_loss = [item["val_loss"] for item in self.history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
        plt.plot(epochs, val_loss, label="Val Loss", color="#d62728", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200, bbox_inches="tight")
        plt.close()


class EarlyStopping:
    """
    Stop training when monitored metric stops improving.
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return True

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
