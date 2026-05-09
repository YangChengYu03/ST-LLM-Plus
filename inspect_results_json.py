"""
Inspect ablation result JSON files and report broken files with context.

Usage:
    python inspect_results_json.py --results_dir ./ablation_results
"""

import argparse
import json
import os
from json import JSONDecodeError


def print_context(path, line_no, radius=3):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as exc:
        print(f"  cannot read context: {exc}")
        return

    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    for idx in range(start, end + 1):
        marker = ">>" if idx == line_no else "  "
        print(f"{marker} {idx:04d}: {lines[idx - 1].rstrip()}")


def main():
    parser = argparse.ArgumentParser(description="Inspect result JSON files")
    parser.add_argument("--results_dir", type=str, default="./ablation_results")
    args = parser.parse_args()

    ok = 0
    broken = 0
    missing = 0

    for exp_name in sorted(os.listdir(args.results_dir)):
        exp_dir = os.path.join(args.results_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        path = os.path.join(exp_dir, "results.json")
        if not os.path.exists(path):
            missing += 1
            print(f"MISSING {path}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
            ok += 1
        except JSONDecodeError as exc:
            broken += 1
            print(f"\nBROKEN {path}")
            print(f"  line {exc.lineno}, column {exc.colno}: {exc.msg}")
            print_context(path, exc.lineno)
        except OSError as exc:
            broken += 1
            print(f"\nBROKEN {path}")
            print(f"  {exc}")

    print(f"\nSummary: ok={ok}, broken={broken}, missing={missing}")


if __name__ == "__main__":
    main()
