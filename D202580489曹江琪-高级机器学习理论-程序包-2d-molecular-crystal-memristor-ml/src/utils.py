from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
SAMPLES = DATA / "samples"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

RANDOM_SEED = 2026

NUMERIC_FEATURES = [
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

FEATURE_LABELS = {
    "bandgap_eV": "带隙",
    "is_metal": "金属性惩罚",
    "defect_trap_score": "缺陷/陷阱态",
    "ion_migration_score": "离子迁移",
    "air_stability_score": "空气稳定性",
    "thermal_stability_score": "热稳定性",
    "fabrication_score": "常规制备成熟度",
    "stm_evidence_score": "STM/STS 证据",
    "surface_order_score": "表面有序度",
    "encapsulation_need_score": "封装需求惩罚",
}

LABEL_MAP = {"low": 0, "medium": 0, "high": 1}

EVIDENCE_LABELS = {
    "high": "已发表锚点",
    "confirmed_mbe": "已验证 MBE 生长候选",
    "suspected_mbe": "待验证 MBE 生长候选",
    "published": "已发表但制备受限",
    "internal": "低优先级内部候选",
}

MATERIAL_DISPLAY = {
    "alpha-P4S5-noncentro": "α-P4S5 非中心相",
    "beta-P4S5-centro": "β-P4S5 中心相",
    "gamma-P4S3I2": "γ-P4S3I2",
    "beta-P4Se3Br2": "β-P4Se3Br2",
    "alpha-As4S4": "α-As4S4",
}


def ensure_dirs() -> None:
    for path in [RAW, SAMPLES, PROCESSED, RESULTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def material_display_name(name: object) -> str:
    text = str(name)
    return MATERIAL_DISPLAY.get(text, text)


def evidence_label(code: object) -> str:
    text = str(code)
    return EVIDENCE_LABELS.get(text, text)


def minmax_fit(df: pd.DataFrame, features: Iterable[str]) -> dict[str, tuple[float, float, float]]:
    stats: dict[str, tuple[float, float, float]] = {}
    for col in features:
        values = pd.to_numeric(df[col], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median)
        lo = float(filled.min())
        hi = float(filled.max())
        stats[col] = (lo, hi, median)
    return stats


def minmax_transform(df: pd.DataFrame, features: Iterable[str], stats: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        lo, hi, median = stats[col]
        values = pd.to_numeric(out[col], errors="coerce").fillna(median)
        if math.isclose(hi, lo):
            out[col] = 0.0
        else:
            out[col] = ((values - lo) / (hi - lo)).clip(0.0, 1.0)
    return out


def safe_numeric(value: object) -> float:
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def normalize_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    lo = values.min()
    hi = values.max()
    if pd.isna(lo) or pd.isna(hi) or math.isclose(float(lo), float(hi)):
        return pd.Series([0.5] * len(values), index=values.index)
    return (values - lo) / (hi - lo)
