from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.decomposition import PCA

from utils import FEATURE_LABELS, FIGURES, PROCESSED, RESULTS, ensure_dirs, read_csv


PALETTE = {
    "blue": "#315f8c",
    "cyan": "#5aa6a8",
    "green": "#4f8b5f",
    "orange": "#d98c3a",
    "red": "#b85c5c",
    "gray": "#6f747a",
    "light": "#eef3f6",
    "grid": "#d9dee3",
}

EVIDENCE_COLORS = {
    "已发表锚点": "#315f8c",
    "已验证 MBE 生长候选": "#4f8b5f",
    "待验证 MBE 生长候选": "#d98c3a",
    "已发表但制备受限": "#b85c5c",
    "低优先级内部候选": "#6f747a",
}


def configure_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ["Microsoft YaHei", "SimHei", "Source Han Sans SC", "Noto Sans CJK SC", "Arial Unicode MS"]:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#222222"
    plt.rcParams["axes.labelcolor"] = "#222222"
    plt.rcParams["xtick.color"] = "#333333"
    plt.rcParams["ytick.color"] = "#333333"
    plt.rcParams["savefig.bbox"] = "tight"


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=360, facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def save_alias(source_stem: str, alias_stem: str) -> None:
    for suffix in ["png", "svg"]:
        source = FIGURES / f"{source_stem}.{suffix}"
        target = FIGURES / f"{alias_stem}.{suffix}"
        if source.exists():
            target.write_bytes(source.read_bytes())


def metric_plot(metrics: pd.DataFrame) -> None:
    metrics = metrics.sort_values("f1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    y = np.arange(len(metrics))
    ax.hlines(y, metrics["f1_mean"], metrics["roc_auc_mean"], color=PALETTE["grid"], lw=2.5, zorder=1)
    ax.scatter(metrics["f1_mean"], y, s=90, color=PALETTE["blue"], label="F1", zorder=3)
    ax.scatter(metrics["roc_auc_mean"], y, s=90, color=PALETTE["orange"], label="AUC", zorder=3)
    ax.scatter(metrics["pr_auc_mean"], y, s=70, color=PALETTE["green"], label="PR-AUC", zorder=3)
    ax.set_yticks(y, metrics["model"])
    lower = max(0.70, float(metrics[["f1_mean", "roc_auc_mean", "pr_auc_mean"]].min().min()) - 0.04)
    ax.set_xlim(lower, 1.02)
    ax.set_xlabel("交叉验证平均指标")
    ax.set_title("模型性能对比", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.8, alpha=0.75)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.70, 1.08))
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig_model_metrics")


def feature_importance_plot(features: pd.DataFrame) -> None:
    features = features.sort_values("importance_normalized", ascending=True).tail(10)
    labels = [FEATURE_LABELS.get(item, item) for item in features["feature"]]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    colors = [PALETTE["blue"] if i >= len(features) - 3 else PALETTE["cyan"] for i in range(len(features))]
    ax.barh(labels, features["importance_normalized"], color=colors, height=0.62)
    for y, value in enumerate(features["importance_normalized"]):
        ax.text(value + 0.006, y, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlabel("置换重要性归一化值")
    ax.set_title("关键物理描述符贡献", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig_feature_importance")


def ranking_plot(ranking: pd.DataFrame) -> None:
    top = ranking.sort_values("rank").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    colors = [EVIDENCE_COLORS.get(label, PALETTE["gray"]) for label in top["evidence_level_cn"]]
    ax.barh(top["material_display"], top["total_score"], color=colors, height=0.62)
    for y, (_, row) in enumerate(top.iterrows()):
        ax.text(row["total_score"] + 0.01, y, f"{row['total_score']:.3f}", va="center", fontsize=9)
        ax.text(0.02, y, f"Top5={row['top5_probability']:.2f}", va="center", color="white", fontsize=8)
    ax.set_xlim(0, max(0.86, float(top["total_score"].max()) + 0.09))
    ax.set_xlabel("综合适配性评分")
    ax.set_title("候选二维无机分子晶体排序", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    handles = []
    for label, color in EVIDENCE_COLORS.items():
        if label in set(ranking["evidence_level_cn"]):
            handles.append(plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=9, label=label))
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    save_figure(fig, "fig_candidate_ranking")


def candidate_radar_plot(ranking: pd.DataFrame) -> None:
    metrics = [
        ("electronic_score", "电子结构"),
        ("defect_score", "缺陷/迁移"),
        ("stability_score", "稳定性"),
        ("fabrication_component", "制备窗口"),
        ("stm_score", "STM/STS"),
    ]
    top = ranking.sort_values("rank").head(5).copy()
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.2, 6.5))
    ax = fig.add_subplot(111, polar=True)
    for idx, (_, row) in enumerate(top.iterrows()):
        values = [float(row[col]) for col, _ in metrics]
        values += values[:1]
        ax.plot(angles, values, lw=2, label=row["material_display"])
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1], [label for _, label in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(color=PALETTE["grid"], lw=0.8)
    ax.set_title("前五名候选材料五维适配雷达图", loc="left", fontsize=15, fontweight="bold", pad=18)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.20), ncols=3, fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.22)
    save_figure(fig, "candidate_radar")


def feature_space_pca_plot(train: pd.DataFrame, candidates: pd.DataFrame, ranking: pd.DataFrame) -> None:
    feature_cols = [
        "bandgap_eV",
        "is_metal",
        "defect_trap_score",
        "ion_migration_score",
        "air_stability_score",
        "thermal_stability_score",
        "fabrication_score",
        "stm_evidence_score",
        "surface_order_score",
        "encapsulation_need_score",
    ]
    train_x = train[feature_cols].to_numpy(dtype=float)
    candidate_x = candidates[feature_cols].to_numpy(dtype=float)
    all_x = np.vstack([train_x, candidate_x])
    coords = PCA(n_components=2, random_state=2026).fit_transform(all_x)
    train_coords = coords[: len(train)]
    candidate_coords = coords[len(train) :]

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    labels = train["target"].to_numpy(dtype=int)
    ax.scatter(
        train_coords[labels == 0, 0],
        train_coords[labels == 0, 1],
        s=55,
        color="#9aa6b2",
        alpha=0.70,
        label="源域低/中适配",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.scatter(
        train_coords[labels == 1, 0],
        train_coords[labels == 1, 1],
        s=65,
        color=PALETTE["blue"],
        alpha=0.78,
        label="源域高适配",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.scatter(
        candidate_coords[:, 0],
        candidate_coords[:, 1],
        s=90,
        marker="D",
        color=PALETTE["orange"],
        alpha=0.86,
        label="目标候选",
        edgecolor="white",
        linewidth=1.0,
    )
    top5 = ranking.sort_values("rank").head(5)[["rank", "material", "material_display", "total_score"]]
    candidate_index = {material: idx for idx, material in enumerate(candidates["material"])}
    offsets = [(26, 18), (30, -22), (-42, 34), (-48, -26), (42, 32)]
    legend_lines = ["Top 5 候选"]
    for offset_idx, (_, row) in enumerate(top5.iterrows()):
        idx = candidate_index.get(row["material"])
        if idx is None:
            continue
        x_value, y_value = candidate_coords[idx]
        rank = int(row["rank"])
        ax.scatter(
            [x_value],
            [y_value],
            s=150,
            marker="D",
            color=PALETTE["orange"],
            edgecolor="#222222",
            linewidth=1.0,
            zorder=4,
        )
        ax.annotate(
            str(rank),
            xy=(x_value, y_value),
            xytext=offsets[offset_idx],
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="circle,pad=0.26", fc=PALETTE["orange"], ec="white", lw=1.0),
            arrowprops=dict(arrowstyle="-", color="#7f8790", lw=0.8, shrinkA=4, shrinkB=6),
            zorder=5,
        )
        legend_lines.append(f"{rank}. {row['material_display']}  {row['total_score']:.3f}")
    ax.text(
        0.02,
        0.04,
        "\n".join(legend_lines),
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=PALETTE["grid"], alpha=0.92),
    )
    ax.axhline(0, color=PALETTE["grid"], lw=0.8)
    ax.axvline(0, color=PALETTE["grid"], lw=0.8)
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.set_title("源域材料与目标候选的特征空间分布", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.13)
    save_figure(fig, "feature_space_pca")


def deposition_window_plot(candidates: pd.DataFrame, ranking: pd.DataFrame) -> None:
    merged = candidates.merge(ranking[["material", "deposition_window_score"]], on="material", how="left")
    plot_df = merged.dropna(subset=["sublimation_temp_c", "decomposition_temp_c"]).copy()
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    max_temp = 330
    ax.fill_between([0, max_temp], [0, max_temp], max_temp, color="#e9f4ef", alpha=0.9, label="可沉积窗口")
    ax.fill_between([0, max_temp], 0, [0, max_temp], color="#f7ece8", alpha=0.9, label="分解风险区")
    ax.plot([0, max_temp], [0, max_temp], color="#333333", lw=1.2, ls="--")
    sizes = 120 + 220 * plot_df["deposition_window_score"].fillna(0.3)
    ax.scatter(
        plot_df["sublimation_temp_c"],
        plot_df["decomposition_temp_c"],
        s=sizes,
        c=plot_df["deposition_window_score"],
        cmap="viridis",
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    for _, row in plot_df.iterrows():
        ax.text(row["sublimation_temp_c"] + 4, row["decomposition_temp_c"] + 3, row["material_display"], fontsize=9)
    ax.set_xlim(0, max_temp)
    ax.set_ylim(0, max_temp)
    ax.set_xlabel("升华温度 / °C")
    ax.set_ylabel("分解温度 / °C")
    ax.set_title("升华-分解沉积窗口", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fig_deposition_window")


def ablation_plot(ablation: pd.DataFrame) -> None:
    top_materials = (
        ablation[ablation["setting"] == "完整特征"]
        .sort_values("rank")
        .head(8)["material_display"]
        .tolist()
    )
    subset = ablation[ablation["material_display"].isin(top_materials)].copy()
    pivot = subset.pivot(index="material_display", columns="setting", values="rank")
    pivot = pivot.loc[top_materials]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    im = ax.imshow(pivot.values, cmap="YlGnBu_r", vmin=1, vmax=16, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", fontsize=10, color="#222222")
    ax.set_title("消融实验中的排名稳定性", loc="left", fontsize=15, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("排名，越小越优")
    ax.tick_params(length=0)
    save_figure(fig, "fig_ablation_uncertainty")


def main() -> None:
    ensure_dirs()
    configure_style()
    metrics = read_csv(RESULTS / "model_metrics.csv")
    features = read_csv(RESULTS / "feature_importance.csv")
    ranking = read_csv(RESULTS / "candidate_ranking.csv")
    ablation = read_csv(RESULTS / "ranking_ablation.csv")
    candidates = read_csv(PROCESSED / "candidate_features.csv")
    train = read_csv(PROCESSED / "train_dataset.csv")

    metric_plot(metrics)
    feature_importance_plot(features)
    ranking_plot(ranking)
    candidate_radar_plot(ranking)
    feature_space_pca_plot(train, candidates, ranking)
    deposition_window_plot(candidates, ranking)
    ablation_plot(ablation)
    save_alias("fig_model_metrics", "model_comparison")
    save_alias("fig_feature_importance", "feature_importance")
    save_alias("fig_ablation_uncertainty", "ranking_ablation")
    print(f"Saved report figures under: {FIGURES}")


if __name__ == "__main__":
    main()
