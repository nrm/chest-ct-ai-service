from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from radiassist.data.test_loader import TestDataLoader
from radiassist.data.luna16_test_loader import LUNA16TestDataLoader
from radiassist.validation.dicom_validator import ChestCTValidator, ValidationResult

import pydicom


def validate_input_data(dicom_dir: Path) -> Tuple[bool, dict, str]:
    """Validate input DICOM data for chest CT analysis."""
    try:
        validator = ChestCTValidator()
        report = validator.validate_dicom_directory(dicom_dir)

        is_valid = report.result == ValidationResult.VALID_CHEST_CT
        confidence = report.confidence

        metadata = {
            "study_uid": "UNKNOWN",
            "series_uid": "UNKNOWN",
            "patient_id": "UNKNOWN",
            "modality": report.modality,
            "num_slices": report.slice_count,
            "body_part": report.body_part,
            "manufacturer": report.manufacturer,
            "validation_result": report.result.value,
            "validation_confidence": confidence,
            "spatial_dimensions": report.spatial_dimensions,
            "spacing": report.spacing
        }

        # Try to extract additional DICOM metadata if validation passed
        if is_valid:
            try:
                dicom_files = [
                    file_path for file_path in dicom_dir.iterdir()
                    if file_path.is_file() and
                    (file_path.suffix == ".dcm" or ("." not in file_path.name and len(file_path.name) >= 4))
                ]

                if dicom_files:
                    ds = pydicom.dcmread(dicom_files[0], force=True)
                    metadata.update({
                        "study_uid": getattr(ds, "StudyInstanceUID", "UNKNOWN"),
                        "series_uid": getattr(ds, "SeriesInstanceUID", "UNKNOWN"),
                        "patient_id": getattr(ds, "PatientID", "UNKNOWN")
                    })
            except Exception as e:
                print(f"    ⚠️  Warning: Could not extract DICOM UIDs: {e}")

        # Create validation message
        if is_valid:
            message = f"✅ Valid chest CT (confidence: {confidence:.3f})"
        else:
            message = f"❌ Invalid data: {report.result.value} (confidence: {confidence:.3f})"
            if report.warnings:
                message += f"\n   Warnings: {'; '.join(report.warnings[:3])}"

        return is_valid, metadata, message

    except Exception as e:
        return False, {"error": str(e)}, f"❌ Validation failed: {e}"


def extract_dicom_metadata(dicom_dir: Path) -> Optional[dict]:
    """Extract key DICOM metadata for reporting purposes."""
    # This function is now replaced by validate_input_data but kept for compatibility
    is_valid, metadata, _ = validate_input_data(dicom_dir)
    return metadata if is_valid else None


def run_covid_triage(model, device: torch.device, dicom_dir: str) -> Tuple[float, str]:
    """Run COVID19 classifier model on given DICOM directory."""
    if model is None:
        return 0.5, "COVID19 model not loaded"

    try:
        loader = TestDataLoader()
        volume = loader.load_dicom_directory(dicom_dir)
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).to(device)

        # Compact mode: comment out verbose logs
        # print(f"    🔢 Input tensor shape: {volume_tensor.shape}")
        # print(
        #     "    📊 Input stats: "
        #     f"min={volume_tensor.min():.3f}, max={volume_tensor.max():.3f}, "
        #     f"mean={volume_tensor.mean():.3f}"
        # )

        with torch.no_grad():
            print("    🧠 Running COVID19 classifier forward pass...")
            # New model returns logits (B,) or (logits, attention_weights)
            outputs = model(volume_tensor)

            # Handle both return formats
            if isinstance(outputs, tuple):
                logits, attention_weights = outputs
                # print(f"    📊 Attention weights shape: {attention_weights.shape}")
            else:
                logits = outputs

            # print(f"    📈 Model logits shape: {logits.shape}")
            # print(f"    🎯 Raw logits: {logits[0].item():.4f}")

            # Apply sigmoid to get probability
            pathology_prob = torch.sigmoid(logits[0]).item()
            print(f"    ✅ Final pathology probability: {pathology_prob:.4f}")

        return pathology_prob, "Success"

    except Exception as exc:  # pragma: no cover - logging path
        print(f"    ❌ COVID19 inference error: {exc}")
        import traceback

        traceback.print_exc()
        return 0.5, f"COVID19 classifier error: {exc}"


def _sample_slices(volume: np.ndarray, n_slices: int = 16) -> np.ndarray:
    """Uniformly sample or pad slices along the Z axis."""
    z_dim = volume.shape[0]

    if z_dim <= n_slices:
        pad_needed = n_slices - z_dim
        if pad_needed > 0:
            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode="edge")
    else:
        indices = np.linspace(0, z_dim - 1, n_slices).astype(int)
        volume = volume[indices]

    return volume


def compute_iou_3d(pos1: tuple, pos2: tuple, patch_size: int = 64) -> float:
    """Compute 3D IoU between two nodule positions (bounding boxes).

    Args:
        pos1, pos2: (z, y, x) positions of detected nodules
        patch_size: Size of detection patch (default 64)

    Returns:
        IoU value [0.0-1.0]
    """
    z1, y1, x1 = pos1
    z2, y2, x2 = pos2

    # Calculate intersection
    z_overlap = max(0, min(z1 + patch_size, z2 + patch_size) - max(z1, z2))
    y_overlap = max(0, min(y1 + patch_size, y2 + patch_size) - max(y1, y2))
    x_overlap = max(0, min(x1 + patch_size, x2 + patch_size) - max(x1, x2))

    intersection = z_overlap * y_overlap * x_overlap

    # Calculate union
    volume_box = patch_size ** 3
    union = 2 * volume_box - intersection

    return intersection / union if union > 0 else 0.0


def apply_nms_3d(detections: list, iou_threshold: float = 0.3, patch_size: int = 64) -> list:
    """Apply Non-Maximum Suppression to remove overlapping detections.

    Args:
        detections: List of dicts with 'position' and 'confidence'
        iou_threshold: IoU threshold for suppression (default 0.3)
        patch_size: Size of detection patch (default 64)

    Returns:
        Filtered list of detections
    """
    if not detections:
        return []

    # Sort by confidence (descending)
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []
    while sorted_dets:
        # Keep highest confidence detection
        best = sorted_dets.pop(0)
        keep.append(best)

        # Remove overlapping detections
        sorted_dets = [
            d for d in sorted_dets
            if compute_iou_3d(best['position'], d['position'], patch_size) < iou_threshold
        ]

    return keep


def filter_nodules_by_size(detections: list, spacing_mm: float = 1.0,
                           min_size_mm: float = 3.0, max_size_mm: float = 35.0,
                           patch_size: int = 64) -> list:
    """Filter nodules by physical size constraints.

    Args:
        detections: List of nodule detections
        spacing_mm: Voxel spacing in mm (default 1.0 for resampled data)
        min_size_mm: Minimum nodule size in mm (default 3mm)
        max_size_mm: Maximum nodule size in mm (default 35mm)
        patch_size: Detection patch size in voxels (default 64)

    Returns:
        Filtered detections (removes patches that are too small/large to be real nodules)
    """
    # Convert patch size to mm
    patch_size_mm = patch_size * spacing_mm

    # For now, we only know patch size, not actual nodule size
    # Simple heuristic: patch should be reasonable size for nodules
    # Metastases: 4-18mm, large nodules: 10-30mm
    # Patch of 64 voxels @ 1mm = 64mm is larger than max nodule
    # So we keep all detections for now (size filtering would need segmentation)

    # Future improvement: estimate nodule size from confidence heatmap
    return detections


def load_nifti_volume(nifti_path: Path, n_slices: int = 16) -> np.ndarray:
    """Load and preprocess a NIfTI volume similarly to training pipeline."""
    nii_img = nib.load(str(nifti_path))
    volume = nii_img.get_fdata().astype(np.float32)

    if volume.ndim == 4:
        volume = volume[..., 0]

    volume = np.transpose(volume, (2, 1, 0))
    volume = np.clip(volume, -1000, 400)
    volume = (volume + 1000) / 1400.0

    volume = _sample_slices(volume, n_slices=n_slices)
    return np.ascontiguousarray(volume)


def run_luna_detection(model, device: torch.device, dicom_dir: str) -> Tuple[dict, str]:
    """Run LUNA16 nodule detection with sliding window."""
    if model is None:
        return {
            'nodule_count': 0,
            'detected_nodules': [],
            'pathology_localization': None,
            'avg_confidence': 0.0,
            'patch_count': 0
        }, "LUNA16 model not loaded"

    try:
        # CRITICAL: For cancer dataset, use FULL resolution (no cropping)
        # This ensures all nodules are visible and coordinates are valid
        # Check if this might be cancer dataset (thin slices)
        from pathlib import Path
        import pydicom

        # Quick check: load one DICOM to check slice thickness
        dicom_dir_path = Path(dicom_dir)
        sample_dicom = None
        for f in dicom_dir_path.rglob("*"):
            if f.is_file():
                try:
                    sample_dicom = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
                    break
                except:
                    continue

        # If thin slices (< 3mm), likely cancer dataset - use full resolution
        slice_thickness = float(getattr(sample_dicom, 'SliceThickness', 5.0)) if sample_dicom else 5.0
        use_full_res = slice_thickness < 3.0

        if use_full_res:
            print(f"    📏 Thin slices detected ({slice_thickness}mm), using FULL resolution for cancer")
            max_slices = 10000  # No practical limit
            resize_inplane = False  # Keep native resolution for cancer
        else:
            print(f"    📏 Thick slices ({slice_thickness}mm), using optimized 160-slice limit")
            max_slices = 160
            resize_inplane = True  # Standard 512x512 for LUNA16

        loader = LUNA16TestDataLoader(max_slices=max_slices, resize_inplane=resize_inplane)
        volume = loader.load_dicom_directory(dicom_dir)

        # print(f"    📐 LUNA16 input volume shape: {volume.shape}")

        patch_size = 64
        stride = 32
        confidence_threshold = 0.25  # Balanced threshold for nodule detection

        # Handle different volume shapes properly
        if volume.ndim == 3:  # (depth, height, width)
            volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        elif volume.ndim == 4:  # (1, depth, height, width) - squeeze first dim
            volume = volume.squeeze(0)  # Remove first dimension
            volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected volume shape: {volume.shape}")

        volume_tensor = volume_tensor.to(device)

        # print(f"    🔢 Volume tensor shape for LUNA16: {volume_tensor.shape}")
        print(f"    🎯 Using FULL RESOLUTION - no additional interpolation needed!")

        # Skip interpolation - LUNA16TestDataLoader already provides optimal resolution
        # Original volume shape is preserved for maximum nodule detection accuracy

        # Ensure we have exactly 5D tensor: (batch, channel, depth, height, width)
        if volume_tensor.ndim != 5:
            raise ValueError(f"Expected 5D tensor, got {volume_tensor.ndim}D: {volume_tensor.shape}")

        depth, height, width = volume_tensor.shape[2:]

        patch_count = 0
        total_confidence = 0.0
        detected_nodules = []

        for z in range(0, depth - patch_size + 1, stride):
            for y in range(0, height - patch_size + 1, stride):
                for x in range(0, width - patch_size + 1, stride):
                    patch = volume_tensor[:, :, z : z + patch_size, y : y + patch_size, x : x + patch_size]
                    if patch.shape[2:] != (patch_size, patch_size, patch_size):
                        continue

                    with torch.no_grad():
                        outputs = model(patch)
                        probabilities = torch.softmax(outputs, dim=1)
                        nodule_prob = probabilities[0, 1].item()

                    total_confidence += nodule_prob
                    patch_count += 1

                    if nodule_prob > confidence_threshold:
                        detected_nodules.append(
                            {
                                "position": (z, y, x),
                                "confidence": nodule_prob,
                                "patch_size": patch_size,
                            }
                        )

        avg_confidence = total_confidence / patch_count if patch_count > 0 else 0.0

        print(f"    🔍 Processed {patch_count} patches")
        print(f"    📊 Average nodule confidence: {avg_confidence:.4f}")
        print(f"    🎯 Raw detections (>{confidence_threshold}): {len(detected_nodules)}")

        # IMPROVEMENT 1: Apply Non-Maximum Suppression to remove overlapping detections
        if detected_nodules:
            detected_nodules = apply_nms_3d(detected_nodules, iou_threshold=0.3, patch_size=patch_size)
            print(f"    ✨ After NMS (IoU<0.3): {len(detected_nodules)} nodules")

        # IMPROVEMENT 2: Size-based filtering (currently no-op, needs segmentation)
        # detected_nodules = filter_nodules_by_size(detected_nodules, spacing_mm=1.0)

        nodule_count = len(detected_nodules)

        # Calculate bounding box for most confident nodules
        pathology_localization = None
        if detected_nodules:
            # Sort by confidence and take top nodules
            sorted_nodules = sorted(detected_nodules, key=lambda x: x['confidence'], reverse=True)
            top_nodules = sorted_nodules[:min(5, len(sorted_nodules))]  # Top 5 most confident

            # Calculate bounding box encompassing top nodules
            z_coords = [n['position'][0] for n in top_nodules]
            y_coords = [n['position'][1] for n in top_nodules]
            x_coords = [n['position'][2] for n in top_nodules]

            # Add patch size to get max coordinates
            z_min, z_max = min(z_coords), max(z_coords) + patch_size
            y_min, y_max = min(y_coords), max(y_coords) + patch_size
            x_min, x_max = min(x_coords), max(x_coords) + patch_size

            pathology_localization = [x_min, x_max, y_min, y_max, z_min, z_max]

            print(f"    📍 Pathology localization: x={x_min}-{x_max}, y={y_min}-{y_max}, z={z_min}-{z_max}")
            print(f"    🏆 Top nodule confidence: {sorted_nodules[0]['confidence']:.3f}")

        return {
            'nodule_count': nodule_count,
            'detected_nodules': detected_nodules,
            'pathology_localization': pathology_localization,
            'avg_confidence': avg_confidence,
            'patch_count': patch_count,
            'volume': volume  # NEW: Return volume for cancer classifier (in [0,1] range)
        }, f"Success (avg_conf: {avg_confidence:.3f}, patches: {patch_count})"

    except Exception as exc:  # pragma: no cover - logging path
        print(f"    ❌ LUNA16 inference error: {exc}")
        import traceback

        traceback.print_exc()
        return {
            'nodule_count': 0,
            'detected_nodules': [],
            'pathology_localization': None,
            'avg_confidence': 0.0,
            'patch_count': 0
        }, f"LUNA16 detection error: {exc}"


def run_ksl_analysis(ksl_analyzer, zip_path: str):
    """Run KSL Z-profile analysis if analyzer is available."""
    if not ksl_analyzer:
        return None, "KSL analyzer not available"

    try:
        print("    🔬 Running KSL Z-profile analysis...")
        result = ksl_analyzer.analyze_zip_file(zip_path)

        if result["available"] and not result["error"]:
            z_score = result["z_profile_score"]
            medical_features = result["medical_features"]

            print(f"    📊 Z-profile score: {z_score:.4f}")
            print(f"    🫁 Avg lung density: {medical_features.get('avg_dense_500', 0):.4f}")
            print(
                f"    🔄 Motion artifacts: {result.get('motion_artifacts', {}).get('avg_correlation', 0):.3f}"
            )

            return result, "Success"

        print(f"    ❌ KSL analysis failed: {result.get('error', 'Unknown error')}")
        return None, result.get("error", "Unknown error")

    except Exception as exc:  # pragma: no cover - logging path
        print(f"    ❌ KSL analysis error: {exc}")
        return None, str(exc)


from .cancer_inference import run_cancer_classification


__all__ = [
    "extract_dicom_metadata",
    "run_covid_triage",
    "run_luna_detection",
    "run_ksl_analysis",
    "run_cancer_classification",  # NEW: Cancer malignancy classification
    "load_nifti_volume",
    "_sample_slices",
]
