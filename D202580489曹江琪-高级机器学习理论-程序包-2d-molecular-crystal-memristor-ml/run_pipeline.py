from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STEPS = [
    ("1/4 构建训练集和候选材料特征", ROOT / "src" / "build_dataset.py"),
    ("2/4 训练并比较机器学习模型", ROOT / "src" / "train_models.py"),
    ("3/4 计算候选材料排序、消融和不确定性", ROOT / "src" / "rank_candidates.py"),
    ("4/4 生成报告用结果图", ROOT / "src" / "plot_results.py"),
]

EXPECTED_OUTPUTS = [
    ROOT / "data" / "samples" / "sample_source_2d_memristor_materials.csv",
    ROOT / "data" / "samples" / "sample_candidate_materials.csv",
    ROOT / "data" / "processed" / "train_dataset.csv",
    ROOT / "data" / "processed" / "candidate_features.csv",
    ROOT / "results" / "model_metrics.csv",
    ROOT / "results" / "model_hyperparameters.csv",
    ROOT / "results" / "feature_importance.csv",
    ROOT / "results" / "candidate_ranking.csv",
    ROOT / "results" / "ranking_ablation.csv",
    ROOT / "results" / "experiment_protocol.json",
    ROOT / "results" / "ranking_protocol.json",
    ROOT / "results" / "figures" / "fig_model_metrics.png",
    ROOT / "results" / "figures" / "fig_model_metrics.svg",
    ROOT / "results" / "figures" / "model_comparison.png",
    ROOT / "results" / "figures" / "model_comparison.svg",
    ROOT / "results" / "figures" / "fig_feature_importance.png",
    ROOT / "results" / "figures" / "feature_importance.png",
    ROOT / "results" / "figures" / "feature_importance.svg",
    ROOT / "results" / "figures" / "fig_candidate_ranking.png",
    ROOT / "results" / "figures" / "candidate_radar.png",
    ROOT / "results" / "figures" / "candidate_radar.svg",
    ROOT / "results" / "figures" / "feature_space_pca.png",
    ROOT / "results" / "figures" / "feature_space_pca.svg",
    ROOT / "results" / "figures" / "fig_deposition_window.png",
    ROOT / "results" / "figures" / "fig_ablation_uncertainty.png",
    ROOT / "results" / "figures" / "ranking_ablation.png",
    ROOT / "results" / "figures" / "ranking_ablation.svg",
]


def run_step(label: str, script: Path) -> None:
    print(f"\n== {label} ==")
    env = os.environ.copy()
    env.setdefault("LOKY_MAX_CPU_COUNT", "2")
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True, env=env)


def main() -> None:
    print(f"Python executable: {sys.executable}")
    for label, script in STEPS:
        run_step(label, script)

    missing = [path for path in EXPECTED_OUTPUTS if not path.exists()]
    if missing:
        print("\n缺少预期输出文件:")
        for path in missing:
            print(f"- {path.relative_to(ROOT)}")
        raise SystemExit(1)

    print("\nPipeline completed successfully. Generated outputs:")
    for path in EXPECTED_OUTPUTS:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
