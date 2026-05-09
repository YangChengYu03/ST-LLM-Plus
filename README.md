# ST-LLM-Plus

ST-LLM-Plus is a traffic congestion forecasting project based on a graph-enhanced spatio-temporal large language model. The project predicts road-segment-level congestion index values with historical traffic sequences, graph priors, GPT-2 + LoRA fine-tuning, and optional multi-source geographic features.

This repository contains the training pipeline, model definitions, ablation experiments, classic baseline comparisons, few-shot experiments, and result collection utilities used for the thesis experiments.

## Features

- ST-LLM+ style traffic forecasting model with GPT-2 backbone and LoRA fine-tuning.
- 1D temporal convolution encoder for historical traffic sequences.
- Prior-knowledge fused graph attention module (PFGA).
- Graph topology attention mask injected into GPT-2 self-attention.
- Optional learnable spatial positional encoding.
- Optional dual-graph fusion with physical distance graph and semantic similarity graph.
- Feature ablation, module ablation, few-shot, original ST-LLM+ comparison, and classic baseline experiments.
- Test-set per-horizon metrics export for MAE, RMSE, MAPE, and WAPE.

## Project Structure

```text
ST-LLM-Plus/
|-- configs/
|   |-- default.py
|   |-- ablation_config.yaml
|   |-- baseline_config.yaml
|   |-- fewshot_config.yaml
|   `-- stllm_plus_comparison_config.yaml
|-- data/
|   |-- dataset.py
|   `-- graph_builder.py
|-- models/
|   |-- st_llm.py
|   |-- pfga.py
|   |-- losses.py
|   `-- baselines.py
|-- utils/
|   |-- logger.py
|   `-- metrics.py
|-- train.py
|-- evaluate.py
|-- run_ablation.py
|-- run_baselines.py
|-- collect_results.py
|-- eval_checkpoints_per_horizon.py
|-- inspect_results_json.py
|-- ranger21.py
`-- requirements.txt
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Transformers
- PEFT
- Matplotlib
- PyYAML

GPU training is strongly recommended because the model uses a GPT-2 backbone.

## Data

The training pipeline expects `.npz` files with a `data` array:

```text
data shape: (T, N, C)
```

where:

- `T`: total time steps
- `N`: number of road segments
- `C`: feature dimension

Default settings:

```text
num_nodes  = 500
input_len  = 12
output_len = 12
feature_dim = 8
```

The graph builder also requires a road coordinate CSV for constructing the physical distance graph.

Default Kaggle paths are configured in the YAML files. If running locally, update:

```yaml
npz_base_dir: "your/local/npz/folder"
coords_csv_path: "your/local/roadsegid_location.csv"
```

## Basic Training

Run the default ST-LLM+ training entry:

```bash
python train.py
```

Important default hyperparameters are defined in:

```text
configs/default.py
```

## Module and Feature Ablation

Run all enabled ablation experiments:

```bash
python run_ablation.py --config configs/ablation_config.yaml
```

Run module ablation only:

```bash
python run_ablation.py --config configs/ablation_config.yaml --batch 1
```

Run feature ablation only:

```bash
python run_ablation.py --config configs/ablation_config.yaml --batch 2
```

Run one experiment:

```bash
python run_ablation.py --config configs/ablation_config.yaml --only mod_no_pfga
```

Results are saved to:

```text
ablation_results/
```

## Classic Baselines

The project includes SGTCN, ASTGCN, and AGCRN baseline implementations.

Run baseline experiments:

```bash
python run_baselines.py --config configs/baseline_config.yaml
```

Run a single baseline:

```bash
python run_baselines.py --config configs/baseline_config.yaml --only agcrn
```

## Few-Shot Experiments

Run few-shot experiments with 5%, 10%, 20%, and 70% training data:

```bash
python run_ablation.py --config configs/fewshot_config.yaml
```

Run only 10% training data:

```bash
python run_ablation.py --config configs/fewshot_config.yaml --only fewshot_10pct
```

Results are saved to:

```text
fewshot_results/
```

## Original ST-LLM+ Comparison

Run original-style ST-LLM+ and improved ST-LLM+ comparison:

```bash
python run_ablation.py --config configs/stllm_plus_comparison_config.yaml
```

In this repository, `original_stllm_plus` is defined as:

- traffic sequence input only
- physical distance graph only
- PFGA enabled
- GPT-2 + LoRA enabled
- graph topology mask enabled
- no multi-source external features
- no learnable spatial positional encoding
- no dual-graph semantic fusion

## Result Collection

Collect experiment results:

```bash
python collect_results.py --results_dir ./ablation_results
```

Generated files include:

```text
summary.md
summary.csv
summary.tex
feature_per_horizon_metrics.csv
feature_per_horizon_metrics_wide.csv
```

If some `results.json` files are corrupted, inspect them with:

```bash
python inspect_results_json.py --results_dir ./ablation_results
```

## Recompute Test Per-Horizon Metrics

If `results.json` is missing or corrupted, recompute test-set per-horizon metrics directly from checkpoints:

```bash
python eval_checkpoints_per_horizon.py \
  --config configs/ablation_config.yaml \
  --results_dir ./ablation_results \
  --batch 1 \
  --output_dir ./recomputed_module_test_metrics
```

For feature ablation:

```bash
python eval_checkpoints_per_horizon.py \
  --config configs/ablation_config.yaml \
  --results_dir ./ablation_results \
  --batch 2 \
  --output_dir ./recomputed_feature_test_metrics
```

Outputs:

```text
test_per_horizon_metrics.csv
test_per_horizon_metrics_wide.csv
test_per_horizon_metrics.json
```

## Key Metrics

The project reports:

- MAE
- RMSE
- MAPE
- WAPE

Per-horizon metrics are usually reported for:

```text
horizon_1, horizon_6, horizon_12
```

## Notes

- Model weights, `.npz` datasets, `.npy` graph files, caches, and experiment outputs are ignored by Git.
- Keep large datasets and checkpoints outside the repository or upload them through external storage.
- For Kaggle experiments, update the YAML config paths to match the mounted dataset paths.
