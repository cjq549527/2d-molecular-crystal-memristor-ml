from __future__ import annotations

import shutil

import pandas as pd

from utils import (
    DATA,
    LABEL_MAP,
    NUMERIC_FEATURES,
    PROCESSED,
    RAW,
    SAMPLES,
    ensure_dirs,
    evidence_label,
    material_display_name,
    minmax_fit,
    minmax_transform,
    read_csv,
    write_csv,
    write_json,
)


def validate_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} 缺少必要字段: {', '.join(missing)}")


def export_input_samples(source: pd.DataFrame, candidates: pd.DataFrame) -> None:
    """Export small visible samples required by the course scoring rubric."""
    sample_source = source.head(8).copy()
    sample_candidates = candidates.head(8).copy()
    write_csv(sample_source, SAMPLES / "sample_source_2d_memristor_materials.csv")
    write_csv(sample_candidates, SAMPLES / "sample_candidate_materials.csv")
    shutil.copyfile(DATA / "candidate_materials.csv", SAMPLES / "full_candidate_materials.csv")


def main() -> None:
    ensure_dirs()
    source = read_csv(RAW / "source_2d_memristor_materials.csv")
    candidates = read_csv(DATA / "candidate_materials.csv")

    source_required = ["material", "family", "memristor_label", "evidence_note"] + NUMERIC_FEATURES
    candidate_required = ["material", "material_class", "evidence_level", "evidence_note"] + NUMERIC_FEATURES
    validate_columns(source, source_required, "source_2d_memristor_materials.csv")
    validate_columns(candidates, candidate_required, "candidate_materials.csv")

    source["target"] = source["memristor_label"].map(LABEL_MAP)
    if source["target"].isna().any():
        missing = sorted(source[source["target"].isna()]["memristor_label"].dropna().unique())
        raise ValueError(f"发现未知标签: {missing}")

    stats = minmax_fit(source, NUMERIC_FEATURES)
    source_scaled = minmax_transform(source, NUMERIC_FEATURES, stats)
    candidate_scaled = minmax_transform(candidates, NUMERIC_FEATURES, stats)

    source_scaled["material_display"] = source_scaled["material"].map(material_display_name)
    candidate_scaled["material_display"] = candidate_scaled["material"].map(material_display_name)
    candidate_scaled["evidence_level_cn"] = candidate_scaled["evidence_level"].map(evidence_label)

    train_columns = ["material", "material_display", "family", "memristor_label", "target"] + NUMERIC_FEATURES + ["evidence_note"]
    candidate_meta_columns = [
        "shg_x_ags",
        "birefringence_delta_n",
        "sublimation_temp_c",
        "decomposition_temp_c",
        "process_note",
    ]
    candidate_meta_columns = [col for col in candidate_meta_columns if col in candidate_scaled.columns]
    candidate_columns = (
        ["material", "material_display", "material_class", "evidence_level", "evidence_level_cn"]
        + NUMERIC_FEATURES
        + ["evidence_note"]
        + candidate_meta_columns
    )

    write_csv(source_scaled[train_columns], PROCESSED / "train_dataset.csv")
    write_csv(candidate_scaled[candidate_columns], PROCESSED / "candidate_features.csv")
    export_input_samples(source, candidates)

    stats_payload = {key: {"min": value[0], "max": value[1], "median_for_missing": value[2]} for key, value in stats.items()}
    write_json(
        {
            "random_seed": 2026,
            "feature_scaling": "Min-max scaling fitted on the source-domain material table.",
            "missing_value_policy": "Missing numeric descriptors are imputed with the source-domain median before scaling.",
            "features": NUMERIC_FEATURES,
            "statistics": stats_payload,
        },
        PROCESSED / "normalization_stats.json",
    )

    print(f"Built training dataset: {PROCESSED / 'train_dataset.csv'} ({len(source_scaled)} rows)")
    print(f"Built candidate dataset: {PROCESSED / 'candidate_features.csv'} ({len(candidate_scaled)} rows)")
    print(f"Exported input samples: {SAMPLES}")


if __name__ == "__main__":
    main()
