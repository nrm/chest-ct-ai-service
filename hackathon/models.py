from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch

from radiassist.models.covid19_classifier import COVID19Classifier

# LUNA16 and Cancer models DISABLED (2025-10-02)
# from radiassist.models.luna16_detector import LUNA16NoduleDetector
# from cancer.models import CancerNoduleClassifier


def load_covid_model(workspace_path: str, device: torch.device) -> Optional[COVID19Classifier]:
    """Load pretrained COVID19 classifier model (ResNet50 MIL)."""
    # Try local API weights first (repo-stored model)
    classifier_path = Path(__file__).parent.parent / "models" / "covid19_classifier_fold1_best_auc.pth"

    # Fallback to legacy workspace layout
    if not classifier_path.exists():
        classifier_path = Path(workspace_path) / "models" / "covid19_classifier_fold1_best_auc.pth"

    if not classifier_path.exists():
        print(f"❌ COVID19 classifier not found at {classifier_path}")
        return None

    # Create model with same architecture as training
    model = COVID19Classifier(
        n_slices=64,
        pretrained=False,  # Weights already trained
        freeze_backbone=False,
        dropout=0.5
    )

    load_kwargs = {"map_location": device}

    try:
        # Try modern PyTorch loading
        checkpoint = torch.load(classifier_path, weights_only=False, **load_kwargs)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(classifier_path, **load_kwargs)
    except Exception as e:
        # Handle numpy compatibility issues
        if "numpy._core" in str(e):
            print(f"⚠️  Numpy compatibility issue, trying with pickle protocol fix...")
            import pickle
            with open(classifier_path, 'rb') as f:
                checkpoint = pickle.load(f)
        else:
            raise e

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✅ COVID19 classifier loaded from {classifier_path.name}")
    print(f"   Architecture: ResNet50 2D MIL (AUC: 0.9839)")
    return model


def load_luna_model(workspace_path: str, device: torch.device) -> None:
    """LUNA16 model DISABLED (2025-10-02) - not used in production."""
    return None


def load_cancer_ensemble(workspace_path: str, device: torch.device) -> None:
    """Cancer classifier DISABLED (2025-10-02) - not used in production."""
    return None


__all__ = ["load_covid_model", "load_luna_model", "load_cancer_ensemble"]
