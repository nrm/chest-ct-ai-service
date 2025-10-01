"""COVID19 classification module."""

from .preprocessing import (
    PreprocessConfig,
    COVID19Preprocessor,
    preprocess_dataset
)

from .data_splits import (
    create_splits,
    save_splits,
    load_splits,
    print_splits_summary,
    verify_splits_integrity
)

from .dataset import COVID19Dataset
from .augmentations import get_train_augmentations
from .losses import FocalLoss, WeightedBCEWithLogitsLoss
from .metrics import compute_metrics, print_metrics, ClassificationMetrics

from .models.classifier import (
    SliceEncoder,
    AttentionPooling,
    COVID19Classifier
)

__all__ = [
    # Preprocessing
    'PreprocessConfig',
    'COVID19Preprocessor',
    'preprocess_dataset',

    # Data splits
    'create_splits',
    'save_splits',
    'load_splits',
    'print_splits_summary',
    'verify_splits_integrity',

    # Dataset and augmentations
    'COVID19Dataset',
    'get_train_augmentations',

    # Models
    'SliceEncoder',
    'AttentionPooling',
    'COVID19Classifier',

    # Loss functions
    'FocalLoss',
    'WeightedBCEWithLogitsLoss',

    # Metrics
    'compute_metrics',
    'print_metrics',
    'ClassificationMetrics'
]