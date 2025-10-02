from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk


def extract_patch_around_nodule(
    volume: np.ndarray,
    nodule_position: Tuple[int, int, int],
    patch_size_mm: int = 40,
    spacing_mm: float = 1.0,
) -> Optional[np.ndarray]:
    """Extract 40×40×40mm patch around detected nodule.

    Args:
        volume: 3D volume (D, H, W) in HU, already at 1mm spacing
        nodule_position: (z, y, x) in voxel coordinates
        patch_size_mm: Patch size in mm (default 40)
        spacing_mm: Spacing in mm (default 1.0, assuming already resampled)

    Returns:
        Patch of shape (40, 40, 40) or None if out of bounds
    """
    z, y, x = nodule_position
    patch_size_voxels = int(patch_size_mm / spacing_mm)
    half_size = patch_size_voxels // 2

    # Calculate bounds
    z_start = max(0, z - half_size)
    z_end = min(volume.shape[0], z + half_size)
    y_start = max(0, y - half_size)
    y_end = min(volume.shape[1], y + half_size)
    x_start = max(0, x - half_size)
    x_end = min(volume.shape[2], x + half_size)

    # Extract patch
    patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]

    # Check if patch is large enough (at least 70% of target size)
    min_size = int(patch_size_voxels * 0.7)
    if (patch.shape[0] < min_size or
        patch.shape[1] < min_size or
        patch.shape[2] < min_size):
        return None

    # Pad to exactly 40x40x40 if needed
    if patch.shape != (patch_size_voxels, patch_size_voxels, patch_size_voxels):
        pad_z = (patch_size_voxels - patch.shape[0]) // 2
        pad_y = (patch_size_voxels - patch.shape[1]) // 2
        pad_x = (patch_size_voxels - patch.shape[2]) // 2

        # Symmetric padding
        patch = np.pad(
            patch,
            ((pad_z, patch_size_voxels - patch.shape[0] - pad_z),
             (pad_y, patch_size_voxels - patch.shape[1] - pad_y),
             (pad_x, patch_size_voxels - patch.shape[2] - pad_x)),
            mode='edge'
        )

    # HU windowing and normalization (MUST match training exactly!)
    # LUNA16 uses [-1024, 400], but training used [-1000, 400]
    # Keep training range for consistency
    patch = np.clip(patch, -1000, 400)
    patch = 2.0 * (patch + 1000) / 1400.0 - 1.0  # Map [-1000, 400] to [-1, 1]

    return patch


def load_dicom_as_sitk(dicom_dir: str) -> sitk.Image:
    """Load DICOM directory as SimpleITK image (raw HU values).

    Returns:
        SimpleITK image with raw HU values (not normalized)
    """
    from pathlib import Path
    import pydicom

    dicom_dir = Path(dicom_dir)

    # Find all DICOM files
    dicom_files = []
    for file_path in dicom_dir.rglob("*"):
        if file_path.is_file() and (
            file_path.suffix.lower() == '.dcm' or
            ('.' not in file_path.name and len(file_path.name) >= 4) or
            file_path.name.startswith(("CT_", "MR_", "CR_", "DX_"))
        ):
            try:
                # Verify it's a valid DICOM
                pydicom.dcmread(str(file_path), force=True, stop_before_pixels=True)
                dicom_files.append(str(file_path))
            except:
                pass

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    # Use SimpleITK to read DICOM series
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    sitk_image = reader.Execute()

    return sitk_image


def run_cancer_classification(
    cancer_models: Optional[List[torch.nn.Module]],
    device: torch.device,
    dicom_dir: str,
    detected_nodules: List[Dict],
    luna_volume: Optional[np.ndarray] = None,
    sensitivity_threshold: float = 0.35,
) -> Tuple[float, Dict, str]:
    """Run cancer malignancy classification on detected nodules.

    Args:
        cancer_models: List of cancer classifier models (ensemble from 5 folds)
        device: Device to use (cuda/cpu)
        dicom_dir: Path to DICOM directory
        detected_nodules: List of nodules from LUNA16 detector
            [{'position': (z, y, x), 'confidence': float, 'patch_size': int}, ...]
        luna_volume: Preprocessed volume from LUNA16 (in [0,1] range, 1mm spacing)
        sensitivity_threshold: Threshold for binary classification (default 0.35 for conservative sensitivity)

    Returns:
        Tuple of (cancer_probability, metadata_dict, status_string)
        - cancer_probability: Study-level cancer probability [0.0-1.0] (max pooling)
        - metadata: Dict with patch_probs, malignant_count, etc.
        - status: Success or error message
    """
    print(f"    🔬 Cancer classification: models={'LOADED' if cancer_models else 'NONE'}, nodules={len(detected_nodules)}")

    if cancer_models is None or len(cancer_models) == 0:
        return 0.0, {
            'available': False,
            'reason': 'Cancer model not loaded'
        }, "Cancer model not available"

    if len(detected_nodules) == 0:
        # No nodules detected → no cancer
        return 0.0, {
            'available': True,
            'num_nodules': 0,
            'reason': 'No nodules detected by LUNA16'
        }, "Success (no nodules)"

    try:
        # Use LUNA16 volume for coordinate compatibility
        # CRITICAL: LUNA16 now clips after resampling → denormalize is correct
        if luna_volume is not None:
            print("    🔬 Using LUNA16 preprocessed volume (denormalizing to HU)...")
            # LUNA16TestDataLoader normalizes as: (HU + 1024) / 1424 → [0, 1]
            # After fix, values are clipped to [0, 1] before denorm
            # Denormalize back to HU: volume * 1424 - 1024
            volume = luna_volume * 1424.0 - 1024.0

            # Handle shape
            if volume.ndim == 4:
                volume = volume.squeeze(0)  # Remove batch dimension

            print(f"    📐 Volume shape: {volume.shape}")
            print(f"    📏 HU range: min={volume.min():.1f}, max={volume.max():.1f}, mean={volume.mean():.1f}")
        else:
            # Load DICOM directly to match training preprocessing (resample in HU space)
            print("    📂 Loading raw DICOM for cancer preprocessing (matches training)...")
            sitk_image = load_dicom_as_sitk(dicom_dir)

            # Resample to 1mm isotropic spacing
            original_spacing = sitk_image.GetSpacing()
            target_spacing = (1.0, 1.0, 1.0)

            if original_spacing != target_spacing:
                print(f"    📏 Resampling from {original_spacing} to {target_spacing} mm...")
                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing(target_spacing)
                resampler.SetSize([
                    int(np.round(sitk_image.GetSize()[i] * (original_spacing[i] / target_spacing[i])))
                    for i in range(3)
                ])
                resampler.SetOutputDirection(sitk_image.GetDirection())
                resampler.SetOutputOrigin(sitk_image.GetOrigin())
                resampler.SetTransform(sitk.Transform())
                resampler.SetDefaultPixelValue(-1024)
                resampler.SetInterpolator(sitk.sitkLinear)
                sitk_image = resampler.Execute(sitk_image)

            volume = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
            print(f"    📐 Volume shape: {volume.shape}")
            print(f"    📏 HU range: min={volume.min():.1f}, max={volume.max():.1f}, mean={volume.mean():.1f}")

        print(f"    🎯 Processing {len(detected_nodules)} detected nodules...")

        # Extract patches around each nodule
        valid_patches = []
        valid_positions = []
        skipped_count = 0

        for i, nodule in enumerate(detected_nodules):
            position = nodule['position']  # (z, y, x)

            patch = extract_patch_around_nodule(volume, position)

            if patch is not None:
                valid_patches.append(patch)
                valid_positions.append(position)
            else:
                skipped_count += 1

        # Summary of extraction
        if skipped_count > 0:
            print(f"    ⚠️  Skipped {skipped_count} nodules (out of bounds)")

        if len(valid_patches) == 0:
            return 0.0, {
                'available': True,
                'num_nodules': len(detected_nodules),
                'valid_patches': 0,
                'reason': 'No valid patches extracted'
            }, "Success (no valid patches)"

        print(f"    ✅ Extracted {len(valid_patches)} valid patches")

        # Convert to torch tensor (B, C=1, D, H, W)
        patches_tensor = torch.from_numpy(np.array(valid_patches)).float()
        patches_tensor = patches_tensor.unsqueeze(1)  # Add channel dimension
        patches_tensor = patches_tensor.to(device)

        print(f"    🔢 Patches tensor shape: {patches_tensor.shape}")

        # Run ensemble prediction with batching to avoid OOM
        print(f"    🧠 Running ensemble prediction ({len(cancer_models)} models)...")

        batch_size = 64  # Process 64 patches at a time to avoid OOM
        num_patches = patches_tensor.shape[0]

        all_probs = []
        with torch.no_grad():
            for model_idx, model in enumerate(cancer_models):
                model.eval()
                model_probs = []

                # Process in batches
                for i in range(0, num_patches, batch_size):
                    batch = patches_tensor[i:i+batch_size]
                    logits = model(batch).squeeze(1)  # (B,)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    model_probs.append(probs)

                # Concatenate batches
                model_probs = np.concatenate(model_probs)
                all_probs.append(model_probs)

        # Average across ensemble
        ensemble_probs = np.mean(all_probs, axis=0)  # (B,)

        print(f"    📊 Patch probabilities: min={ensemble_probs.min():.3f}, "
              f"max={ensemble_probs.max():.3f}, mean={ensemble_probs.mean():.3f}")

        # Study-level aggregation: simple max pooling (NO weighted pooling)
        # Weighted pooling was destroying probabilities (0.7 → 0.27)
        cancer_probability = float(np.max(ensemble_probs))

        # Count malignant nodules (above threshold)
        malignant_count = int(np.sum(ensemble_probs >= sensitivity_threshold))

        print(f"    🎯 Study-level cancer probability: {cancer_probability:.4f}")
        print(f"    🔴 Malignant nodules (≥{sensitivity_threshold}): {malignant_count}/{len(valid_patches)}")

        # Binary prediction (for screening)
        cancer_prediction = 1 if cancer_probability >= sensitivity_threshold else 0
        prediction_str = "MALIGNANT" if cancer_prediction == 1 else "BENIGN"

        print(f"    🏥 Cancer prediction: {prediction_str}")

        metadata = {
            'available': True,
            'num_nodules_detected': len(detected_nodules),
            'num_patches_processed': len(valid_patches),
            'cancer_probability': cancer_probability,
            'cancer_prediction': cancer_prediction,
            'malignant_count': malignant_count,
            'patch_probabilities': ensemble_probs.tolist(),
            'patch_positions': valid_positions,
            'threshold': sensitivity_threshold,
            'ensemble_size': len(cancer_models),
        }

        status = f"Success (prob: {cancer_probability:.3f}, {prediction_str})"

        return cancer_probability, metadata, status

    except Exception as exc:
        print(f"    ❌ Cancer classification error: {exc}")
        import traceback
        traceback.print_exc()

        return 0.0, {
            'available': True,
            'error': str(exc),
            'reason': 'Cancer classification failed'
        }, f"Cancer classification error: {exc}"


__all__ = [
    "run_cancer_classification",
    "extract_patch_around_nodule",
]
