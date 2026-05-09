"""
ST-LLM+ 消融实验结果收集器

独立脚本，用于收集所有实验结果并生成对比表格。
可在任意时刻运行，无需 GPU。

使用方式：
    python collect_results.py                                  # 默认路径
    python collect_results.py --results_dir ./ablation_results  # 自定义路径

输出：
    - summary.md  : Markdown 格式汇总表（方便查看）
    - summary.csv : CSV 格式（方便导入 Excel/LaTeX）
"""

import os
import json
import argparse
import time
from json import JSONDecodeError


def collect_all_results(results_dir):
    """遍历所有实验目录，收集 results.json"""
    all_results = {}

    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在: {results_dir}")
        return all_results

    for exp_name in sorted(os.listdir(results_dir)):
        exp_dir = os.path.join(results_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        results_path = os.path.join(exp_dir, 'results.json')
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    all_results[exp_name] = json.load(f)
            except JSONDecodeError as e:
                print(
                    f"⚠️ 跳过损坏的 results.json: {results_path} "
                    f"(line {e.lineno}, column {e.colno}: {e.msg})"
                )
            except OSError as e:
                print(f"⚠️ 无法读取 results.json: {results_path} ({e})")

    return all_results


def generate_markdown_table(all_results, output_path):
    """生成 Markdown 汇总表"""
    lines = []
    lines.append("# ST-LLM+ 消融实验结果汇总\n")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ====== 总体指标表 ======
    lines.append("## 总体指标\n")
    lines.append("| 实验名 | 说明 | MAE ↓ | RMSE ↓ | MAPE(%) ↓ | WAPE(%) ↓ | 训练时间(min) | Best Epoch |")
    lines.append("|--------|------|-------|--------|-----------|-----------|---------------|------------|")

    # 先按类型分组：feat_ 开头 / mod_ 开头
    feat_exps = {k: v for k, v in all_results.items() if k.startswith('feat_')}
    mod_exps = {k: v for k, v in all_results.items() if k.startswith('mod_')}
    other_exps = {k: v for k, v in all_results.items()
                  if not k.startswith('feat_') and not k.startswith('mod_')}

    def add_rows(experiments, lines):
        for exp_name, results in experiments.items():
            m = results['metrics']['overall']
            t = results['training']
            desc = results.get('description', '')
            lines.append(
                f"| {exp_name} | {desc} | "
                f"{m['mae']:.4f} | {m['rmse']:.4f} | "
                f"{m['mape']:.2f} | {m['wape']:.2f} | "
                f"{t['total_train_time_sec']/60:.1f} | {t['best_epoch']} |"
            )

    if feat_exps:
        lines.append("")
        lines.append("**特征消融实验：**\n")
        lines.append("| 实验名 | 说明 | MAE ↓ | RMSE ↓ | MAPE(%) ↓ | WAPE(%) ↓ | 训练时间(min) | Best Epoch |")
        lines.append("|--------|------|-------|--------|-----------|-----------|---------------|------------|")
        add_rows(feat_exps, lines)

    if mod_exps:
        lines.append("")
        lines.append("**模块消融实验：**\n")
        lines.append("| 实验名 | 说明 | MAE ↓ | RMSE ↓ | MAPE(%) ↓ | WAPE(%) ↓ | 训练时间(min) | Best Epoch |")
        lines.append("|--------|------|-------|--------|-----------|-----------|---------------|------------|")
        add_rows(mod_exps, lines)

    if other_exps:
        lines.append("")
        lines.append("**其他实验：**\n")
        lines.append("| 实验名 | 说明 | MAE ↓ | RMSE ↓ | MAPE(%) ↓ | WAPE(%) ↓ | 训练时间(min) | Best Epoch |")
        lines.append("|--------|------|-------|--------|-----------|-----------|---------------|------------|")
        add_rows(other_exps, lines)

    # ====== 分步长指标表 ======
    all_horizons = set()
    for results in all_results.values():
        all_horizons.update(results['metrics'].get('per_horizon', {}).keys())

    if all_horizons:
        sorted_horizons = sorted(all_horizons)

        for metric_name in ['mae', 'rmse']:
            lines.append(f"\n## 分步长 {metric_name.upper()}\n")
            header = "| 实验名 |"
            sep = "|--------|"
            for h in sorted_horizons:
                header += f" {h} |"
                sep += "------|"
            lines.append(header)
            lines.append(sep)

            for exp_name, results in all_results.items():
                ph = results['metrics'].get('per_horizon', {})
                row = f"| {exp_name} |"
                for h in sorted_horizons:
                    if h in ph:
                        row += f" {ph[h][metric_name]:.4f} |"
                    else:
                        row += " - |"
                lines.append(row)

    # ====== 消融增量分析 ======
    if 'feat_full' in all_results:
        lines.append("\n## 消融增量分析（相对于 feat_full 基准）\n")
        baseline = all_results['feat_full']['metrics']['overall']
        lines.append("| 实验名 | ΔMAE | ΔRMSE | ΔMAPE(%) | ΔWAPE(%) | 性能变化 |")
        lines.append("|--------|------|-------|----------|----------|----------|")

        for exp_name, results in all_results.items():
            if exp_name == 'feat_full':
                continue
            m = results['metrics']['overall']
            d_mae = m['mae'] - baseline['mae']
            d_rmse = m['rmse'] - baseline['rmse']
            d_mape = m['mape'] - baseline['mape']
            d_wape = m['wape'] - baseline['wape']

            # 性能变化判定
            if d_mae > 0:
                change = f"📉 下降 (MAE +{d_mae:.4f})"
            elif d_mae < 0:
                change = f"📈 提升 (MAE {d_mae:.4f})"
            else:
                change = "➡️ 持平"

            lines.append(
                f"| {exp_name} | "
                f"{d_mae:+.4f} | {d_rmse:+.4f} | "
                f"{d_mape:+.2f} | {d_wape:+.2f} | {change} |"
            )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return lines


def generate_csv(all_results, output_path):
    """生成 CSV 汇总文件"""
    lines = ["experiment,description,MAE,RMSE,MAPE,WAPE,train_time_min,best_epoch,num_params,num_trainable"]

    for exp_name, results in all_results.items():
        m = results['metrics']['overall']
        t = results['training']
        desc = results.get('description', '').replace(',', ';')
        lines.append(
            f"{exp_name},{desc},"
            f"{m['mae']:.6f},{m['rmse']:.6f},"
            f"{m['mape']:.4f},{m['wape']:.4f},"
            f"{t['total_train_time_sec']/60:.1f},{t['best_epoch']},"
            f"{t.get('num_params', '')},{t.get('num_trainable_params', '')}"
        )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def generate_per_horizon_csv(all_results, output_path, only_feature=True):
    """Generate long-format per-horizon metrics CSV.

    Columns:
        experiment,description,horizon,MAE,RMSE,MAPE,WAPE

    This is intended for feature-ablation step-wise comparison and can be
    imported directly into Excel/SPSS/Origin.
    """
    lines = ["experiment,description,horizon,MAE,RMSE,MAPE,WAPE"]

    for exp_name, results in all_results.items():
        if only_feature and not exp_name.startswith("feat_"):
            continue

        desc = results.get('description', '').replace(',', ';')
        per_horizon = results.get('metrics', {}).get('per_horizon', {})

        def horizon_key(name):
            try:
                return int(str(name).split("_")[-1])
            except ValueError:
                return 10 ** 9

        for horizon in sorted(per_horizon.keys(), key=horizon_key):
            metric = per_horizon[horizon]
            lines.append(
                f"{exp_name},{desc},{horizon},"
                f"{metric.get('mae', ''):.6f},"
                f"{metric.get('rmse', ''):.6f},"
                f"{metric.get('mape', ''):.4f},"
                f"{metric.get('wape', ''):.4f}"
            )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def generate_per_horizon_wide_csv(all_results, output_path, only_feature=True):
    """Generate wide-format per-horizon metrics CSV.

    Each experiment occupies one row. Columns are horizon_1_MAE,
    horizon_1_RMSE, horizon_1_MAPE, horizon_1_WAPE, etc.
    """
    filtered = {
        name: result for name, result in all_results.items()
        if (not only_feature or name.startswith("feat_"))
    }

    horizons = set()
    for result in filtered.values():
        horizons.update(result.get('metrics', {}).get('per_horizon', {}).keys())

    def horizon_key(name):
        try:
            return int(str(name).split("_")[-1])
        except ValueError:
            return 10 ** 9

    sorted_horizons = sorted(horizons, key=horizon_key)
    metrics = ["mae", "rmse", "mape", "wape"]

    header = ["experiment", "description"]
    for horizon in sorted_horizons:
        for metric in metrics:
            header.append(f"{horizon}_{metric.upper()}")

    lines = [",".join(header)]
    for exp_name, results in filtered.items():
        row = [exp_name, results.get('description', '').replace(',', ';')]
        per_horizon = results.get('metrics', {}).get('per_horizon', {})
        for horizon in sorted_horizons:
            horizon_metrics = per_horizon.get(horizon, {})
            for metric in metrics:
                value = horizon_metrics.get(metric, "")
                row.append("" if value == "" else f"{value:.6f}")
        lines.append(",".join(row))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def generate_latex_table(all_results, output_path):
    """生成 LaTeX 表格（方便论文直接使用）"""
    lines = []
    lines.append("% ST-LLM+ 消融实验结果 LaTeX 表格")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation Study Results of ST-LLM+}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & MAE $\downarrow$ & RMSE $\downarrow$ & MAPE(\%) $\downarrow$ & WAPE(\%) $\downarrow$ \\")
    lines.append(r"\midrule")

    # 基准行加粗
    for exp_name, results in all_results.items():
        m = results['metrics']['overall']
        desc = results.get('description', exp_name)

        if exp_name == 'feat_full':
            lines.append(
                f"\\textbf{{{desc}}} & \\textbf{{{m['mae']:.4f}}} & "
                f"\\textbf{{{m['rmse']:.4f}}} & "
                f"\\textbf{{{m['mape']:.2f}}} & "
                f"\\textbf{{{m['wape']:.2f}}} \\\\"
            )
        else:
            lines.append(
                f"{desc} & {m['mae']:.4f} & "
                f"{m['rmse']:.4f} & "
                f"{m['mape']:.2f} & "
                f"{m['wape']:.2f} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description="ST-LLM+ 消融实验结果收集器")
    parser.add_argument("--results_dir", type=str, default="./ablation_results",
                        help="实验结果根目录")
    args = parser.parse_args()

    print(f"📂 扫描结果目录: {args.results_dir}")

    all_results = collect_all_results(args.results_dir)

    if not all_results:
        print("⚠️ 未找到任何实验结果！")
        return

    print(f"✅ 找到 {len(all_results)} 个实验结果:")
    for name in all_results:
        print(f"   - {name}")

    # 生成输出
    md_path = os.path.join(args.results_dir, "summary.md")
    csv_path = os.path.join(args.results_dir, "summary.csv")
    per_horizon_csv_path = os.path.join(args.results_dir, "feature_per_horizon_metrics.csv")
    per_horizon_wide_csv_path = os.path.join(args.results_dir, "feature_per_horizon_metrics_wide.csv")
    tex_path = os.path.join(args.results_dir, "summary.tex")

    md_lines = generate_markdown_table(all_results, md_path)
    generate_csv(all_results, csv_path)
    generate_per_horizon_csv(all_results, per_horizon_csv_path, only_feature=True)
    generate_per_horizon_wide_csv(all_results, per_horizon_wide_csv_path, only_feature=True)
    generate_latex_table(all_results, tex_path)

    print(f"\n📊 输出文件:")
    print(f"   Markdown: {md_path}")
    print(f"   CSV:      {csv_path}")
    print(f"   Step CSV: {per_horizon_csv_path}")
    print(f"   Step Wide CSV: {per_horizon_wide_csv_path}")
    print(f"   LaTeX:    {tex_path}")

    # 打印到控制台
    print("\n" + "=" * 100)
    for line in md_lines:
        print(line)
    print("=" * 100)


if __name__ == "__main__":
    main()
