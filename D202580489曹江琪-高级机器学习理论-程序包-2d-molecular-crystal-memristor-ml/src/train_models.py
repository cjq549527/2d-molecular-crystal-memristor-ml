from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from utils import NUMERIC_FEATURES, PROCESSED, RANDOM_SEED, RESULTS, ensure_dirs, read_csv, write_csv, write_json


@dataclass
class ModelSpec:
    name: str
    estimator: object
    category: str
    rationale: str


def model_specs() -> list[ModelSpec]:
    """Define baseline and representative modern tabular-learning models."""
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_SEED)
    svm = SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=RANDOM_SEED)
    gnb = GaussianNB()
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.055,
        max_leaf_nodes=7,
        min_samples_leaf=2,
        l2_regularization=0.04,
        random_state=RANDOM_SEED,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(12, 6),
        activation="relu",
        solver="lbfgs",
        alpha=0.05,
        max_iter=2500,
        random_state=RANDOM_SEED,
    )
    ensemble = VotingClassifier(
        estimators=[
            ("lr", clone(lr)),
            ("rf", clone(rf)),
            ("et", clone(et)),
            ("hgb", clone(hgb)),
            ("mlp", clone(mlp)),
        ],
        voting="soft",
        weights=[1.0, 1.2, 1.2, 1.1, 1.0],
    )
    return [
        ModelSpec("Logistic Regression", lr, "基础线性模型", "可解释的线性分类基线。"),
        ModelSpec("Linear SVM", svm, "基础间隔模型", "通过最大间隔边界检验线性可分性。"),
        ModelSpec("Gaussian Naive Bayes", gnb, "概率基线模型", "在小样本条件下提供低方差概率基线。"),
        ModelSpec("Random Forest", rf, "树集成模型", "通过自助采样和特征随机化降低单棵树方差。"),
        ModelSpec("Extra Trees", et, "强随机树集成模型", "使用更强随机切分增强小样本稳定性。"),
        ModelSpec("HistGradientBoosting", hgb, "梯度提升树模型", "以逐步拟合残差的方式捕捉非线性特征组合。"),
        ModelSpec("MLP", mlp, "浅层神经网络", "通过非线性隐层学习材料描述符之间的组合关系。"),
        ModelSpec("Soft Voting Ensemble", ensemble, "校准集成模型", "融合线性、树模型和神经网络输出，降低单模型偶然性。"),
    ]


def predict_scores(estimator: object, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        return np.asarray(proba[:, 1], dtype=float)
    decision = estimator.decision_function(x)
    decision = np.asarray(decision, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -40, 40)))


def metric_row(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pred = (scores >= 0.5).astype(int)
    row = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, scores),
        "brier": brier_score_loss(y_true, scores),
    }
    try:
        row["roc_auc"] = roc_auc_score(y_true, scores)
    except ValueError:
        row["roc_auc"] = 0.5
    return row


def summarize(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    ci95 = float(1.96 * std / np.sqrt(max(len(arr), 1)))
    return mean, std, ci95


def evaluate_models(x: np.ndarray, y: np.ndarray, specs: list[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=20, test_size=0.30, random_state=RANDOM_SEED)
    metric_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []

    for spec in specs:
        repeated_metrics: list[dict[str, float]] = []
        for split_id, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
            estimator = clone(spec.estimator)
            estimator.fit(x[train_idx], y[train_idx])
            scores = predict_scores(estimator, x[test_idx])
            repeated_metrics.append(metric_row(y[test_idx], scores))
            for idx, score in zip(test_idx, scores):
                prediction_records.append(
                    {
                        "model": spec.name,
                        "split": split_id,
                        "row_index": int(idx),
                        "true_label": int(y[idx]),
                        "predicted_score": round(float(score), 6),
                    }
                )

        row: dict[str, object] = {
            "model": spec.name,
            "category": spec.category,
            "rationale": spec.rationale,
            "n_splits": len(repeated_metrics),
        }
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]:
            mean, std, ci95 = summarize([m[metric] for m in repeated_metrics])
            row[f"{metric}_mean"] = round(mean, 4)
            row[f"{metric}_std"] = round(std, 4)
            row[f"{metric}_ci95"] = round(ci95, 4)
        metric_records.append(row)

    metrics = pd.DataFrame(metric_records).sort_values(["f1_mean", "roc_auc_mean", "pr_auc_mean"], ascending=False)
    predictions = pd.DataFrame(prediction_records)
    return metrics, predictions


def single_model_importance(spec: ModelSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    estimator = clone(spec.estimator)
    estimator.fit(x, y)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.ravel(np.abs(estimator.coef_)).astype(float)
    else:
        result = permutation_importance(
            estimator,
            x,
            y,
            scoring="roc_auc",
            n_repeats=40,
            random_state=RANDOM_SEED,
        )
        values = np.asarray(result.importances_mean, dtype=float).clip(min=0)
    total = float(np.abs(values).sum())
    if total <= 0:
        return np.zeros(len(NUMERIC_FEATURES), dtype=float)
    return np.abs(values) / total


def compute_feature_importance(specs: list[ModelSpec], x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    usable = []
    for spec in specs:
        values = single_model_importance(spec, x, y)
        if values.sum() > 0:
            usable.append(values)
            for feature, value in zip(NUMERIC_FEATURES, values):
                rows.append({"model": spec.name, "feature": feature, "model_importance": value})
    matrix = np.vstack(usable) if usable else np.zeros((1, len(NUMERIC_FEATURES)))
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    importance = pd.DataFrame(
        {
            "feature": NUMERIC_FEATURES,
            "importance_mean": means,
            "importance_std": stds,
        }
    )
    total = float(importance["importance_mean"].sum())
    if total > 0:
        importance["importance_normalized"] = importance["importance_mean"] / total
    else:
        importance["importance_normalized"] = 0.0
    detail = pd.DataFrame(rows)
    write_csv(detail, RESULTS / "feature_importance_by_model.csv")
    return importance.sort_values("importance_normalized", ascending=False)


def hyperparameter_table(specs: list[ModelSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        params = spec.estimator.get_params(deep=False)
        compact = []
        for key in ["C", "kernel", "n_estimators", "max_depth", "max_iter", "learning_rate", "hidden_layer_sizes", "alpha"]:
            if key in params:
                compact.append(f"{key}={params[key]}")
        rows.append(
            {
                "model": spec.name,
                "category": spec.category,
                "main_hyperparameters": "; ".join(compact) if compact else "default",
                "rationale": spec.rationale,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    df = read_csv(PROCESSED / "train_dataset.csv")
    x = df[NUMERIC_FEATURES].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)

    specs = model_specs()
    metrics, predictions = evaluate_models(x, y, specs)
    best_name = str(metrics.iloc[0]["model"])
    best_spec = next(spec for spec in specs if spec.name == best_name)
    importance = compute_feature_importance(specs, x, y)
    hyperparameters = hyperparameter_table(specs)

    write_csv(metrics, RESULTS / "model_metrics.csv")
    write_csv(predictions, RESULTS / "model_predictions.csv")
    write_csv(importance, RESULTS / "feature_importance.csv")
    write_csv(hyperparameters, RESULTS / "model_hyperparameters.csv")
    write_json(
        {
            "random_seed": RANDOM_SEED,
            "validation_scheme": "20 repeated stratified holdout splits",
            "test_fraction": 0.30,
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"],
            "best_model": best_name,
            "boundary": "Metrics evaluate internal consistency of a curated small-sample screening dataset; they are not direct device-performance proof.",
        },
        RESULTS / "experiment_protocol.json",
    )

    print(f"Saved metrics: {RESULTS / 'model_metrics.csv'}")
    print(f"Best model: {best_name}")
    print(f"Saved feature importance: {RESULTS / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
