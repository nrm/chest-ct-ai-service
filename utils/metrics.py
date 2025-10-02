"""Statistical metrics with confidence intervals for medical AI evaluation."""

import numpy as np
from typing import Tuple


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for binomial proportion.

    More accurate than normal approximation for small sample sizes.

    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if trials == 0:
        return (0.0, 0.0)

    from scipy import stats

    p = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)

    return (lower, upper)


def calculate_metrics_with_ci(y_true: np.ndarray, y_pred: np.ndarray, confidence: float = 0.95) -> dict:
    """Calculate classification metrics with 95% confidence intervals.

    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        confidence: Confidence level (default: 0.95)

    Returns:
        Dictionary with metrics and their confidence intervals
    """
    # Confusion matrix
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n = len(y_true)
    n_positive = tp + fn
    n_negative = tn + fp

    # Point estimates
    accuracy = (tp + tn) / n if n > 0 else 0
    sensitivity = tp / n_positive if n_positive > 0 else 0
    specificity = tn / n_negative if n_negative > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Confidence intervals using Wilson score method
    acc_ci = wilson_score_interval(tp + tn, n, confidence)
    sens_ci = wilson_score_interval(tp, n_positive, confidence) if n_positive > 0 else (0.0, 0.0)
    spec_ci = wilson_score_interval(tn, n_negative, confidence) if n_negative > 0 else (0.0, 0.0)
    prec_ci = wilson_score_interval(tp, tp + fp, confidence) if (tp + fp) > 0 else (0.0, 0.0)

    return {
        'n_samples': n,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'confusion_matrix': {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        },
        'accuracy': {
            'value': accuracy,
            'ci_lower': acc_ci[0],
            'ci_upper': acc_ci[1],
            'ci_range': f'({acc_ci[0]:.3f}, {acc_ci[1]:.3f})'
        },
        'sensitivity': {
            'value': sensitivity,
            'ci_lower': sens_ci[0],
            'ci_upper': sens_ci[1],
            'ci_range': f'({sens_ci[0]:.3f}, {sens_ci[1]:.3f})'
        },
        'specificity': {
            'value': specificity,
            'ci_lower': spec_ci[0],
            'ci_upper': spec_ci[1],
            'ci_range': f'({spec_ci[0]:.3f}, {spec_ci[1]:.3f})'
        },
        'precision': {
            'value': precision,
            'ci_lower': prec_ci[0],
            'ci_upper': prec_ci[1],
            'ci_range': f'({prec_ci[0]:.3f}, {prec_ci[1]:.3f})'
        }
    }


def print_metrics_report(metrics: dict, dataset_name: str = "Dataset"):
    """Pretty print metrics with confidence intervals.

    Args:
        metrics: Dictionary from calculate_metrics_with_ci()
        dataset_name: Name of dataset for the report
    """
    print(f"\n{'='*70}")
    print(f"📊 {dataset_name} - Diagnostic Metrics with 95% Confidence Intervals")
    print(f"{'='*70}")

    print(f"\n📋 Sample Size:")
    print(f"   Total:     {metrics['n_samples']}")
    print(f"   Positive:  {metrics['n_positive']}")
    print(f"   Negative:  {metrics['n_negative']}")

    cm = metrics['confusion_matrix']
    print(f"\n📐 Confusion Matrix:")
    print(f"   TP: {cm['tp']:<4} FP: {cm['fp']:<4}")
    print(f"   FN: {cm['fn']:<4} TN: {cm['tn']:<4}")

    acc = metrics['accuracy']
    sens = metrics['sensitivity']
    spec = metrics['specificity']
    prec = metrics['precision']

    print(f"\n📈 Metrics (95% CI):")
    print(f"   Accuracy:    {acc['value']:.3f} {acc['ci_range']}")
    print(f"   Sensitivity: {sens['value']:.3f} {sens['ci_range']}")
    print(f"   Specificity: {spec['value']:.3f} {spec['ci_range']}")
    print(f"   Precision:   {prec['value']:.3f} {prec['ci_range']}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Example usage
    print("Example: LCT dataset (n=3, perfect classification)")
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 1])

    metrics = calculate_metrics_with_ci(y_true, y_pred)
    print_metrics_report(metrics, "LCT Dataset")

    print("\nExample: COVID19 dataset (n=60, realistic performance)")
    # Simulate COVID19 results: Acc=50%, Sens=60%, Spec=40%
    np.random.seed(42)
    y_true = np.array([1]*30 + [0]*30)
    y_pred = np.array([1]*18 + [0]*12 + [1]*18 + [0]*12)

    metrics = calculate_metrics_with_ci(y_true, y_pred)
    print_metrics_report(metrics, "COVID19 Dataset")
