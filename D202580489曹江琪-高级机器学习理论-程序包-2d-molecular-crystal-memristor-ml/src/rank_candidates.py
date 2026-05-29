from __future__ import annotations

import math

import numpy as np
import pandas as pd

from utils import PROCESSED, RANDOM_SEED, RESULTS, ensure_dirs, read_csv, safe_numeric, write_csv, write_json


BASE_WEIGHTS = {
    "electronic_score": 0.25,
    "defect_score": 0.25,
    "stability_score": 0.20,
    "fabrication_component": 0.15,
    "stm_score": 0.15,
}


def electronic_score(row: pd.Series) -> float:
    bandgap = float(row["bandgap_eV"])
    is_metal = float(row["is_metal"])
    gap_component = 1.0 - abs(bandgap - 0.52) / 0.52
    gap_component = float(np.clip(gap_component, 0.0, 1.0))
    return 0.75 * gap_component + 0.25 * (1.0 - is_metal)


def deposition_window_score(row: pd.Series) -> tuple[float, str]:
    """Separate thermal stability from depositability.

    A thermally stable molecule is not automatically suitable for thermal
    evaporation. The practical window is favorable only when sublimation can
    occur below decomposition under the available vacuum condition. MBE can
    lower the effective sublimation temperature through higher vacuum and a
    controlled molecular beam, so confirmed MBE candidates are not scored in
    the same way as ordinary thermal evaporation candidates.
    """
    material = str(row.get("material", ""))
    evidence = str(row.get("evidence_level", ""))
    note = f"{row.get('process_note', '')} {row.get('evidence_note', '')}".lower()
    sublimation = safe_numeric(row.get("sublimation_temp_c"))
    decomposition = safe_numeric(row.get("decomposition_temp_c"))

    if material == "Sb2O3":
        return 0.76, "已发表二维无机分子晶体锚点，适合优先作为流程标定材料。"

    if evidence == "confirmed_mbe":
        if "raw-material synthesis remains difficult" in note:
            return 0.62, "已验证可 MBE 生长，但原材料合成仍是近期器件验证的主要风险。"
        if "bulk crystal synthesis conditions still need exploration" in note:
            return 0.42, "列入 MBE 候选，但大量晶体合成条件仍需探索。"
        if not math.isnan(sublimation) and not math.isnan(decomposition):
            margin = decomposition - sublimation
            if margin >= 50:
                return 0.95, "升华温度明显低于分解温度，MBE 沉积窗口清晰。"
            if margin > 0 and sublimation <= 230:
                return 0.78, "升华温度低于分解温度，但窗口较窄，建议优先在 MBE 条件下验证。"
            if margin > 0:
                return 0.68, "可 MBE 生长但所需升华温度较高，工艺窗口需要精细控制。"
        return 0.58, "已有 MBE 候选依据，但升华-分解窗口的定量数据仍需补充。"

    if evidence == "suspected_mbe":
        if "difficult to obtain by sublimation" in note:
            return 0.14, "疑似可 MBE 生长，但热蒸镀/低真空升华路径风险较高。"
        if "phase transition" in note:
            return 0.38, "存在相变或升华窗口不确定性，需先验证高真空沉积窗口。"
        if "higher thermal stability" in note:
            return 0.36, "热稳定性较高但升华窗口未知，不能直接等同于可沉积。"
        if "series properties similar" in note:
            return 0.34, "系列性质类似 α-P4S5，但 MBE 沉积窗口仍待实验确认。"
        return 0.30, "缺少直接升华-分解窗口证据，暂按 MBE 待验证处理。"

    if "cannot be obtained by sublimation" in note:
        return 0.05, "低压下无法升华得到，热蒸镀窗口差，器件验证应暂缓。"
    if "hygroscopic" in note or "cannot exist stably in air" in note:
        return 0.03, "吸潮且空气中不能稳定存在，需先解决封装或原位器件工艺。"

    if not math.isnan(sublimation) and not math.isnan(decomposition):
        margin = decomposition - sublimation
        if margin >= 50:
            return 0.88, "升华温度低于分解温度，沉积窗口较明确。"
        if margin > 0:
            return 0.46, "升华温度仅略低于分解温度，更适合 MBE 条件下验证。"
        return 0.08, "升华温度不低于分解温度，热蒸镀不可行。"

    return 0.26, "公开或实验数据不足，需先补充升华和分解温度。"


def score_candidates(
    df: pd.DataFrame,
    include_stm: bool = True,
    include_fabrication: bool = True,
    include_stability: bool = True,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = weights or BASE_WEIGHTS
    out = df.copy()
    out["electronic_score"] = out.apply(electronic_score, axis=1)
    out["defect_score"] = 0.60 * out["defect_trap_score"] + 0.40 * out["ion_migration_score"]
    out["stability_score"] = (
        0.50 * out["air_stability_score"]
        + 0.35 * out["thermal_stability_score"]
        + 0.15 * (1.0 - out["encapsulation_need_score"])
    )

    deposition = out.apply(deposition_window_score, axis=1)
    out["deposition_window_score"] = [item[0] for item in deposition]
    out["fabrication_note"] = [item[1] for item in deposition]
    out["fabrication_component"] = out["deposition_window_score"] if include_fabrication else 0.50
    if not include_stability:
        out["stability_score"] = 0.50

    out["stm_score"] = (
        0.60 * out["stm_evidence_score"] + 0.40 * out["surface_order_score"] if include_stm else 0.50
    )
    out["total_score"] = sum(weights[key] * out[key] for key in weights)
    out["rank"] = out["total_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["rank", "material_display"])


def recommendation(row: pd.Series) -> str:
    if row["total_score"] >= 0.72 and row["fabrication_component"] >= 0.70:
        return "优先验证：建议进入 MBE 生长、原位 STM/STS 和两端器件测试。"
    if row["fabrication_component"] < 0.20:
        return "暂缓器件：先解决升华/分解窗口、空气稳定性或原位封装问题。"
    if row["fabrication_component"] < 0.45:
        return "工艺待验证：先确认高真空下升华温度是否低于分解温度。"
    if row["stability_score"] < 0.45:
        return "谨慎验证：优先考虑封装或原位器件工艺。"
    return "候选验证：建议补充 STS、DFT 和沉积窗口数据。"


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.01) for value in weights.values())
    return {key: max(value, 0.01) / total for key, value in weights.items()}


def rank_stability(df: pd.DataFrame, n_iter: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rank_records: dict[str, list[int]] = {str(material): [] for material in df["material"]}
    for _ in range(n_iter):
        sampled = {
            key: float(np.clip(value + rng.normal(0, 0.035), 0.04, 0.45))
            for key, value in BASE_WEIGHTS.items()
        }
        scored = score_candidates(df, weights=normalize_weights(sampled))
        for _, row in scored.iterrows():
            rank_records[str(row["material"])].append(int(row["rank"]))

    rows = []
    for material, ranks in rank_records.items():
        arr = np.asarray(ranks, dtype=float)
        rows.append(
            {
                "material": material,
                "rank_mean": round(float(arr.mean()), 3),
                "rank_std": round(float(arr.std(ddof=0)), 3),
                "top5_probability": round(float((arr <= 5).mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    df = read_csv(PROCESSED / "candidate_features.csv")
    stability = rank_stability(df)
    ranked = score_candidates(df).merge(stability, on="material", how="left")
    ranked["recommendation"] = ranked.apply(recommendation, axis=1)

    columns = [
        "rank",
        "material",
        "material_display",
        "material_class",
        "electronic_score",
        "defect_score",
        "stability_score",
        "fabrication_component",
        "deposition_window_score",
        "stm_score",
        "total_score",
        "rank_mean",
        "rank_std",
        "top5_probability",
        "evidence_level_cn",
        "recommendation",
        "fabrication_note",
        "evidence_note",
    ]
    rounded = ranked[columns].copy()
    score_cols = [
        "electronic_score",
        "defect_score",
        "stability_score",
        "fabrication_component",
        "deposition_window_score",
        "stm_score",
        "total_score",
    ]
    rounded[score_cols] = rounded[score_cols].round(4)
    write_csv(rounded, RESULTS / "candidate_ranking.csv")

    ablations = []
    settings = {
        "完整特征": {},
        "去除 STM/STS": {"include_stm": False},
        "去除沉积窗口": {"include_fabrication": False},
        "去除稳定性": {"include_stability": False},
        "仅本征特征": {"include_stm": False, "include_fabrication": False, "include_stability": False},
    }
    for name, kwargs in settings.items():
        scored = score_candidates(df, **kwargs)
        for _, row in scored.iterrows():
            ablations.append(
                {
                    "setting": name,
                    "material": row["material"],
                    "material_display": row["material_display"],
                    "rank": int(row["rank"]),
                    "total_score": round(float(row["total_score"]), 4),
                }
            )
    write_csv(pd.DataFrame(ablations), RESULTS / "ranking_ablation.csv")
    write_json(
        {
            "base_weights": BASE_WEIGHTS,
            "rank_stability_iterations": 500,
            "interpretation": "top5_probability is the probability of remaining in the top five under random perturbation of scoring weights.",
        },
        RESULTS / "ranking_protocol.json",
    )

    print(f"Saved candidate ranking: {RESULTS / 'candidate_ranking.csv'}")
    print(f"Saved ablation ranking: {RESULTS / 'ranking_ablation.csv'}")


if __name__ == "__main__":
    main()
