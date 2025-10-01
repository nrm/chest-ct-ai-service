#!/usr/bin/env python3
"""
Метрики для валидации COVID19 классификатора.

Включает:
- AUC ROC с 95% bootstrap CI
- Sensitivity / Specificity с 95% CI
- Confusion matrix utilities
- Optimal threshold finding
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve
)
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    """Контейнер для метрик классификации."""
    auc_roc: float
    auc_roc_ci_low: float
    auc_roc_ci_high: float

    sensitivity: float
    sensitivity_ci_low: float
    sensitivity_ci_high: float

    specificity: float
    specificity_ci_low: float
    specificity_ci_high: float

    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int

    def __str__(self):
        return (
            f"AUC ROC: {self.auc_roc:.4f} [95% CI: {self.auc_roc_ci_low:.4f}-{self.auc_roc_ci_high:.4f}]\n"
            f"Sensitivity: {self.sensitivity:.4f} [95% CI: {self.sensitivity_ci_low:.4f}-{self.sensitivity_ci_high:.4f}]\n"
            f"Specificity: {self.specificity:.4f} [95% CI: {self.specificity_ci_low:.4f}-{self.specificity_ci_high:.4f}]\n"
            f"Threshold: {self.threshold:.4f}\n"
            f"Confusion matrix: TN={self.tn}, FP={self.fp}, FN={self.fn}, TP={self.tp}"
        )


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Вычисляет bootstrap 95% confidence interval для метрики.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        metric_fn: Функция метрики (y_true, y_pred) -> score
        n_bootstraps: Число bootstrap итераций
        ci: Confidence level (default 95%)
        seed: Random seed

    Returns:
        (metric, ci_low, ci_high)
    """
    np.random.seed(seed)

    n = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Skip if only one class present
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except:
            continue

    if len(bootstrap_scores) == 0:
        return np.nan, np.nan, np.nan

    bootstrap_scores = np.array(bootstrap_scores)

    # Original metric
    metric = metric_fn(y_true, y_pred)

    # Confidence interval
    alpha = 1 - ci
    ci_low = np.percentile(bootstrap_scores, alpha / 2 * 100)
    ci_high = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

    return metric, ci_low, ci_high


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    target_specificity: float = 0.95
) -> float:
    """
    Находит порог вероятности для достижения target specificity.

    Args:
        y_true: Ground truth labels (0=normal, 1=pathology)
        y_pred_proba: Predicted probabilities of pathology
        target_specificity: Целевая специфичность нормы

    Returns:
        optimal_threshold: Порог вероятности
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # Specificity = 1 - FPR
    specificities = 1 - fpr

    # Находим ближайший порог для target_specificity
    idx = np.argmin(np.abs(specificities - target_specificity))

    return thresholds[idx]


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = None,
    target_specificity: float = 0.95,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> ClassificationMetrics:
    """
    Вычисляет полный набор метрик с 95% CI.

    Args:
        y_true: Ground truth labels (0=normal, 1=pathology)
        y_pred_proba: Predicted probabilities of pathology
        threshold: Порог классификации. Если None, находится автоматически
        target_specificity: Целевая специфичность для автоматического threshold
        n_bootstraps: Число bootstrap итераций для CI
        seed: Random seed

    Returns:
        ClassificationMetrics object
    """
    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_pred_proba, target_specificity)

    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Basic metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC ROC with bootstrap CI
    auc_roc, auc_ci_low, auc_ci_high = bootstrap_ci(
        y_true, y_pred_proba,
        metric_fn=roc_auc_score,
        n_bootstraps=n_bootstraps,
        seed=seed
    )

    # Sensitivity with bootstrap CI
    def sens_fn(y_t, y_p):
        y_pred_bin = (y_p >= threshold).astype(int)
        cm = confusion_matrix(y_t, y_pred_bin).ravel()
        if len(cm) == 4:
            _, _, fn, tp = cm
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 0.0

    _, sens_ci_low, sens_ci_high = bootstrap_ci(
        y_true, y_pred_proba,
        metric_fn=sens_fn,
        n_bootstraps=n_bootstraps,
        seed=seed
    )

    # Specificity with bootstrap CI
    def spec_fn(y_t, y_p):
        y_pred_bin = (y_p >= threshold).astype(int)
        cm = confusion_matrix(y_t, y_pred_bin).ravel()
        if len(cm) == 4:
            tn, fp, _, _ = cm
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0

    _, spec_ci_low, spec_ci_high = bootstrap_ci(
        y_true, y_pred_proba,
        metric_fn=spec_fn,
        n_bootstraps=n_bootstraps,
        seed=seed
    )

    return ClassificationMetrics(
        auc_roc=auc_roc,
        auc_roc_ci_low=auc_ci_low,
        auc_roc_ci_high=auc_ci_high,
        sensitivity=sensitivity,
        sensitivity_ci_low=sens_ci_low,
        sensitivity_ci_high=sens_ci_high,
        specificity=specificity,
        specificity_ci_low=spec_ci_low,
        specificity_ci_high=spec_ci_high,
        threshold=threshold,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp)
    )


def print_metrics(metrics: ClassificationMetrics, title: str = "METRICS"):
    """Pretty print метрик."""
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"\nAUC ROC: {metrics.auc_roc:.4f} "
          f"[95% CI: {metrics.auc_roc_ci_low:.4f}-{metrics.auc_roc_ci_high:.4f}]")
    print(f"\nAt threshold = {metrics.threshold:.4f}:")
    print(f"  Sensitivity (recall): {metrics.sensitivity:.4f} "
          f"[95% CI: {metrics.sensitivity_ci_low:.4f}-{metrics.sensitivity_ci_high:.4f}]")
    print(f"  Specificity:          {metrics.specificity:.4f} "
          f"[95% CI: {metrics.specificity_ci_low:.4f}-{metrics.specificity_ci_high:.4f}]")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Normal  Pathology")
    print(f"  Actual Normal    {metrics.tn:4d}    {metrics.fp:4d}")
    print(f"  Actual Pathology {metrics.fn:4d}    {metrics.tp:4d}")
    print("=" * 80)


def test_metrics():
    """Тест вычисления метрик."""
    print("=" * 80)
    print("ТЕСТ METRICS")
    print("=" * 80)

    # Синтетические данные: 100 примеров (25 норма, 75 патология)
    np.random.seed(42)

    n_normal = 25
    n_pathology = 75

    # Ground truth
    y_true = np.array([0] * n_normal + [1] * n_pathology)

    # Predicted probabilities: хорошая модель с некоторым шумом
    # Нормы: mean=0.2, std=0.15 (low probabilities)
    # Патологии: mean=0.8, std=0.15 (high probabilities)
    y_pred_proba = np.concatenate([
        np.clip(np.random.normal(0.2, 0.15, n_normal), 0, 1),
        np.clip(np.random.normal(0.8, 0.15, n_pathology), 0, 1)
    ])

    print(f"\nТестовые данные:")
    print(f"  Total samples: {len(y_true)}")
    print(f"  Normal: {n_normal} ({n_normal/len(y_true)*100:.1f}%)")
    print(f"  Pathology: {n_pathology} ({n_pathology/len(y_true)*100:.1f}%)")
    print(f"\nPredicted probabilities:")
    print(f"  Normal mean: {y_pred_proba[:n_normal].mean():.4f}")
    print(f"  Pathology mean: {y_pred_proba[n_normal:].mean():.4f}")

    # Вычисляем метрики
    print(f"\nВычисление метрик (target_specificity=0.95, n_bootstraps=1000)...")
    metrics = compute_metrics(
        y_true,
        y_pred_proba,
        threshold=None,
        target_specificity=0.95,
        n_bootstraps=1000,
        seed=42
    )

    # Выводим
    print_metrics(metrics, title="РЕЗУЛЬТАТЫ")

    # Проверки
    print("\n" + "=" * 80)
    print("ПРОВЕРКИ")
    print("=" * 80)

    checks = []
    checks.append(("AUC ROC > 0.5", metrics.auc_roc > 0.5))
    checks.append(("AUC ROC CI не включает 0.5", metrics.auc_roc_ci_low > 0.5))
    checks.append(("Specificity близка к 0.95", abs(metrics.specificity - 0.95) < 0.1))
    checks.append(("Sensitivity > 0.5", metrics.sensitivity > 0.5))
    checks.append(("TN + FP = n_normal", metrics.tn + metrics.fp == n_normal))
    checks.append(("FN + TP = n_pathology", metrics.fn + metrics.tp == n_pathology))

    all_passed = True
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")
        if not check_result:
            all_passed = False

    if all_passed:
        print("\n✅ Все проверки пройдены!")
    else:
        print("\n⚠️ Некоторые проверки не прошли")

    print("=" * 80)


if __name__ == "__main__":
    test_metrics()