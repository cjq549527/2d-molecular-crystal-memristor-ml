# 2D Molecular Crystal Memristor ML

本程序用于复现课程报告《面向忆阻器应用的二维无机分子晶体机器学习筛选与性能预测》中的机器学习实验结果。程序输出的是“候选材料实验优先级”，不是已经由真实忆阻器测试证明的最终器件性能排名。

## 1. 环境要求

- Python 3.10 或更高版本
- Windows、macOS 或 Linux
- 依赖库见 `requirements.txt`

安装依赖：

```bash
python -m pip install -r requirements.txt
```

如果 Windows 终端无法识别 `python`，请安装 Python 并勾选 “Add Python to PATH”，或者直接双击 `run_windows.bat`。
注：若本地环境运行时间较长，可直接查看程序包内已生成的 results 目录；报告中的所有表格和程序图均来自该目录。

## 2. 一键运行

在程序包根目录运行：

```bash
python run_pipeline.py
```

Windows 用户也可以双击：

```text
run_windows.bat
```

程序会依次完成数据集构建、模型训练、候选排序和结果图生成。

## 3. 输入样例与来源

主要输入文件如下：

- `data/raw/source_2d_memristor_materials.csv`：源域二维/超薄忆阻器材料样本。来源为公开二维材料忆阻器文献、二维材料数据库和课程报告整理的弱监督标签。
- `data/candidate_materials.csv`：目标域二维无机分子晶体候选材料。来源为公开文献、用户提供的课程材料表和实验先验。
- `data/samples/`：运行 `python run_pipeline.py` 后自动导出的输入样例，用于满足课程评分中“提供测试输入样例并说明来源”的要求。

一个候选材料输入样例如下：

```text
material = PAs3S3
bandgap_eV = 2.75
shg_x_ags = 8.0
birefringence_delta_n = 0.150
sublimation_temp_c = 120
decomposition_temp_c = 207
evidence_level = confirmed_mbe
```

字段解释和数据来源见 `DATA_SOURCES.md` 与 `references.csv`。

## 4. 输出文件

运行完成后会生成：

| 文件 | 内容 | 报告对应位置 |
| --- | --- | --- |
| `data/processed/train_dataset.csv` | 归一化后的源域训练数据 | 数据描述 |
| `data/processed/candidate_features.csv` | 归一化后的候选材料特征 | 数据描述 |
| `results/model_metrics.csv` | Accuracy、Precision、Recall、F1、AUC、PR-AUC 等模型指标 | 模型性能对比 |
| `results/model_hyperparameters.csv` | 模型类别和主要超参数 | 方法与实验设置 |
| `results/feature_importance.csv` | 置换特征重要性 | 特征贡献分析 |
| `results/candidate_ranking.csv` | 16 种候选材料排序和实验建议 | 候选材料排序 |
| `results/ranking_ablation.csv` | 消融实验排序变化 | 消融分析 |
| `results/figures/model_comparison.png` | 模型 F1、AUC 和 PR-AUC 对比 | 图 4 |
| `results/figures/feature_importance.png` | 置换特征重要性 | 图 5 |
| `results/figures/feature_space_pca.png` | 源域样本与目标候选的 PCA 特征空间 | 图 6 |
| `results/figures/candidate_radar.png` | 前五名候选材料雷达图 | 图 7 |
| `results/figures/fig_deposition_window.png` | 升华-分解沉积窗口图 | 图 8 |
| `results/figures/ranking_ablation.png` | 消融实验排名变化 | 图 9 |
| `results/figures/*.svg` | 与 PNG 对应的矢量版本 | 报告备用图 |

## 5. 预期关键结果

正常运行后，`results/candidate_ranking.csv` 的前五名通常为：

```text
1. Sb2O3
2. PAs3S3
3. P3SbS3
4. α-P4S5 非中心相
5. AsP3S5
```

模型性能的最佳项由交叉验证结果自动决定，以 `results/model_metrics.csv` 为准。不同 scikit-learn 版本可能导致小数末位略有差异，但主要排序和结论应保持一致。

## 6. 程序结构

```text
2d-molecular-crystal-memristor-ml/
├─ data/
│  ├─ raw/
│  ├─ samples/
│  ├─ processed/
│  └─ candidate_materials.csv
├─ src/
│  ├─ build_dataset.py
│  ├─ train_models.py
│  ├─ rank_candidates.py
│  ├─ plot_results.py
│  └─ utils.py
├─ results/
│  ├─ *.csv
│  └─ figures/
├─ run_pipeline.py
├─ run_windows.bat
├─ requirements.txt
├─ DATA_SOURCES.md
├─ references.csv
└─ README.md
```

## 7. 常见问题

**依赖安装失败**  
请先确认网络能访问 Python 包索引，然后运行：

```bash
python -m pip install -r requirements.txt
```
