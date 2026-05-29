# 课程程序提交说明

本文件面向评分教师，说明如何复现报告中的程序结果。

## 1. 提交内容

请提交整个 `2d-molecular-crystal-memristor-ml` 程序包，而不是单个 `.py` 文件。程序包至少应包含：

- `src/`：核心 Python 源码；
- `data/`：源域数据、候选材料数据和运行后导出的输入样例；
- `results/`：程序运行后生成的 CSV 和结果图；
- `requirements.txt`：依赖库；
- `README.md`：运行说明；
- `DATA_SOURCES.md` 与 `references.csv`：数据来源和文献说明；
- `run_pipeline.py` 与 `run_windows.bat`：一键运行入口。

## 2. 运行环境

推荐环境：

```text
Python >= 3.10
numpy
pandas
scikit-learn
matplotlib
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 3. 复现命令

在程序包根目录运行：

```bash
python run_pipeline.py
```

Windows 用户可双击 `run_windows.bat`。该脚本会自动安装依赖并运行完整流程。

## 4. 输出检查

运行成功后应看到以下文件：

- `results/model_metrics.csv`
- `results/model_hyperparameters.csv`
- `results/feature_importance.csv`
- `results/candidate_ranking.csv`
- `results/ranking_ablation.csv`
- `results/figures/fig_model_metrics.png`
- `results/figures/model_comparison.png`
- `results/figures/fig_candidate_ranking.png`
- `results/figures/fig_feature_importance.png`
- `results/figures/feature_importance.png`
- `results/figures/feature_space_pca.png`
- `results/figures/candidate_radar.png`
- `results/figures/fig_deposition_window.png`
- `results/figures/fig_ablation_uncertainty.png`
- `results/figures/ranking_ablation.png`

报告中的模型性能、候选排序、消融实验和程序结果图均来自上述文件。

## 5. 结果边界

本程序使用课程报告整理的小样本材料数据，目标是建立可复现的机器学习筛选流程。输出结果用于辅助安排 MBE 生长、STM/STS 表征和忆阻器验证实验，不应被解释为已经完成真实器件验证的最终性能结论。
