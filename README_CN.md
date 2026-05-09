# ST-LLM-Plus 中文说明

ST-LLM-Plus 是一个面向城市道路短时拥堵指数预测的时空大语言模型实验项目。项目基于图增强时空大语言模型思想，结合历史交通序列、图结构先验、GPT-2 主干网络、LoRA 参数高效微调以及多源地理特征，实现道路路段级拥堵指数多步预测。

本仓库包含论文实验所需的模型训练、评估、消融实验、经典基线对比、少量样本实验、原版 ST-LLM+ 对比实验以及结果汇总脚本。

## 项目特点

- 基于 GPT-2 主干网络和 LoRA 微调的 ST-LLM+ 风格交通预测模型。
- 使用 1D 时间卷积编码器提取历史拥堵序列局部模式。
- 引入先验知识融合图注意力模块（PFGA）建模道路空间依赖。
- 将图拓扑注意力掩码注入 GPT-2 自注意力层。
- 支持可学习空间位置编码。
- 支持物理距离图与语义相似图的双图融合。
- 支持模块消融、特征消融、少量样本实验、经典模型对比和原版 ST-LLM+ 对比实验。
- 支持重新加载 checkpoint，在测试集上导出不同预测步长的 MAE、RMSE、MAPE 和 WAPE。

## 项目结构

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

## 环境配置

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Transformers
- PEFT
- Matplotlib
- PyYAML

由于模型使用 GPT-2 主干网络，建议使用 GPU 进行训练。

## 数据格式

训练数据使用 `.npz` 文件，文件中需要包含 `data` 数组：

```text
data shape: (T, N, C)
```

其中：

- `T` 表示总时间步数
- `N` 表示道路路段节点数
- `C` 表示输入特征维度

默认设置为：

```text
num_nodes = 500
input_len = 12
output_len = 12
feature_dim = 8
```

图构建模块还需要道路坐标 CSV 文件，用于构建物理距离图。

如果在本地运行，需要修改 YAML 配置文件中的数据路径：

```yaml
npz_base_dir: "你的本地 npz 数据目录"
coords_csv_path: "你的 roadsegid_location.csv 路径"
```

## 基础训练

运行默认训练入口：

```bash
python train.py
```

默认超参数位于：

```text
configs/default.py
```

## 消融实验

运行全部启用的消融实验：

```bash
python run_ablation.py --config configs/ablation_config.yaml
```

只运行模块消融：

```bash
python run_ablation.py --config configs/ablation_config.yaml --batch 1
```

只运行特征消融：

```bash
python run_ablation.py --config configs/ablation_config.yaml --batch 2
```

运行指定实验：

```bash
python run_ablation.py --config configs/ablation_config.yaml --only mod_no_pfga
```

实验结果默认保存到：

```text
ablation_results/
```

## 经典基线模型实验

项目中实现了 SGTCN、ASTGCN 和 AGCRN 三类经典交通预测基线模型。

运行全部基线实验：

```bash
python run_baselines.py --config configs/baseline_config.yaml
```

只运行某一个基线模型：

```bash
python run_baselines.py --config configs/baseline_config.yaml --only agcrn
```

## 少量样本实验

运行 5%、10%、20% 和 70% 训练数据比例实验：

```bash
python run_ablation.py --config configs/fewshot_config.yaml
```

只运行 10% 训练样本实验：

```bash
python run_ablation.py --config configs/fewshot_config.yaml --only fewshot_10pct
```

实验结果默认保存到：

```text
fewshot_results/
```

## 原版 ST-LLM+ 对比实验

运行原版 ST-LLM+ 风格复现实验与本文改进模型对比：

```bash
python run_ablation.py --config configs/stllm_plus_comparison_config.yaml
```

在本项目中，`original_stllm_plus` 表示：

- 仅使用历史交通序列输入
- 仅使用物理距离图
- 保留 1D 时间卷积
- 保留 PFGA
- 保留 GPT-2 + LoRA
- 保留图拓扑注意力掩码
- 不使用多源外部地理特征
- 不使用可学习空间位置编码
- 不使用语义图与物理图的双图融合

`improved_stllm_plus` 表示本文改进模型，包含多源特征、空间位置编码和双图融合机制。

## 结果汇总

汇总实验结果：

```bash
python collect_results.py --results_dir ./ablation_results
```

会生成：

```text
summary.md
summary.csv
summary.tex
feature_per_horizon_metrics.csv
feature_per_horizon_metrics_wide.csv
```

如果某些 `results.json` 文件损坏，可以使用：

```bash
python inspect_results_json.py --results_dir ./ablation_results
```

## 重新计算测试集分步长指标

如果 `results.json` 缺失或损坏，可以直接读取各实验的 `best_model.pth`，重新在测试集上计算不同预测步长的指标。

模块消融：

```bash
python eval_checkpoints_per_horizon.py \
  --config configs/ablation_config.yaml \
  --results_dir ./ablation_results \
  --batch 1 \
  --output_dir ./recomputed_module_test_metrics
```

特征消融：

```bash
python eval_checkpoints_per_horizon.py \
  --config configs/ablation_config.yaml \
  --results_dir ./ablation_results \
  --batch 2 \
  --output_dir ./recomputed_feature_test_metrics
```

输出文件包括：

```text
test_per_horizon_metrics.csv
test_per_horizon_metrics_wide.csv
test_per_horizon_metrics.json
```

## 评价指标

项目主要使用以下指标评价模型性能：

- MAE
- RMSE
- MAPE
- WAPE

默认统计以下预测步长：

```text
horizon_1
horizon_6
horizon_12
```

## 注意事项

- 模型权重、`.npz` 数据集、`.npy` 图文件、缓存文件和实验输出结果不会提交到 Git。
- 大型数据集和模型 checkpoint 建议放在仓库外部或使用云盘、Kaggle Dataset 等方式管理。
- 在 Kaggle 上运行时，需要根据实际挂载路径修改 YAML 配置文件中的数据路径。

