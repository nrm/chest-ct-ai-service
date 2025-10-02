from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from radiassist.models.covid19_classifier import COVID19Classifier
from radiassist.models.luna16_detector import LUNA16NoduleDetector


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

    # # Check if it's a Git LFS pointer file (small size) - DISABLED FOR REAL MODELS
    # file_size = classifier_path.stat().st_size
    # if file_size < 1000:  # Less than 1KB - likely a Git LFS pointer
    #     print(f"⚠️  WARNING: Model file is {file_size} bytes - appears to be a Git LFS pointer!")
    #     print(f"   Creating mock model for testing. Download real model with: git lfs pull")
    #     model = COVID19Classifier(
    #         n_slices=64,
    #         pretrained=False,
    #         freeze_backbone=False,
    #         dropout=0.5
    #     )
    #     model.to(device)
    #     model.eval()
    #     return model

    print(f"📦 Loading real COVID19 model from: {classifier_path}")
    print(f"📊 Model file size: {classifier_path.stat().st_size / (1024*1024):.2f} MB")
    
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
            print(f"❌ Failed to load model: {e}")
            print(f"   Using untrained model for testing")
            model.to(device)
            model.eval()
            return model

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✅ COVID19 classifier loaded from {classifier_path.name}")
    print(f"   Architecture: ResNet50 2D MIL (AUC: 0.9839)")
    return model


def load_luna_model(workspace_path: str, device: torch.device) -> Optional[LUNA16NoduleDetector]:
    """Load the most recent LUNA16 detector model if available."""
    # Try local API models first, then fallback to workspace
    local_model_dir = Path(__file__).parent.parent / "models"
    workspace_model_dir = Path(workspace_path) / "models"

    # Check local models first
    local_luna_models = sorted(local_model_dir.glob("luna16_detector_*.pth"))
    workspace_luna_models = sorted(workspace_model_dir.glob("luna16_detector_*.pth"))

    luna_models = local_luna_models if local_luna_models else workspace_luna_models

    if not luna_models:
        print("❌ LUNA16 model not found")
        return None

    latest_model = luna_models[-1]
    print(f"🔍 Loading LUNA16 model: {latest_model}")

    # # Check if it's a Git LFS pointer file - DISABLED FOR REAL MODELS
    # file_size = latest_model.stat().st_size
    # if file_size < 1000:
    #     print(f"⚠️  WARNING: LUNA16 model file is {file_size} bytes - appears to be a Git LFS pointer!")
    #     print(f"   Creating mock detector for testing")
    #     detector = LUNA16NoduleDetector(
    #         in_channels=1,
    #         num_classes=2,
    #         feature_maps=[32, 64, 128, 256],
    #         use_attention=True,
    #         use_checkpoint=False,
    #     )
    #     detector.to(device)
    #     detector.eval()
    #     return detector

    print(f"📦 Loading real LUNA16 model: {latest_model.name}")
    print(f"📊 Model file size: {latest_model.stat().st_size / (1024*1024):.2f} MB")

    detector = LUNA16NoduleDetector(
        in_channels=1,
        num_classes=2,
        feature_maps=[32, 64, 128, 256],
        use_attention=True,
        use_checkpoint=False,
    )

    load_kwargs = {"map_location": device}
    try:
        checkpoint = torch.load(latest_model, weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(latest_model, **load_kwargs)
    except Exception as e:
        print(f"❌ Failed to load LUNA16 model: {e}")
        print(f"   Using untrained detector for testing")
        detector.to(device)
        detector.eval()
        return detector

    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.to(device)
    detector.eval()
    print("✅ LUNA16 model loaded successfully")
    return detector


__all__ = ["load_covid_model", "load_luna_model"]
