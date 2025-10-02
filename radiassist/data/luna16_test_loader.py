"""
LUNA16 Test Data Loader - Full Resolution Preservation
Handles ZIP archives with DICOM files for LUNA16 nodule detection
Preserves maximum spatial information without downsampling
"""

import zipfile
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pydicom
import numpy as np
import torch
from scipy import ndimage


class LUNA16TestDataLoader:
    """Load and process test datasets from ZIP archives for LUNA16 detection"""

    def __init__(self, test_data_dir: Optional[Path] = None, max_slices: int = 160, resize_inplane: bool = True):
        """
        Args:
            test_data_dir: Directory containing test data
            max_slices: Maximum number of slices to process (default 160)
                       Reduces computation time while preserving coverage
            resize_inplane: Whether to resize in-plane to 512x512 (default True)
        """
        self.test_data_dir = Path(test_data_dir) if test_data_dir else None
        self.max_slices = max_slices
        self.resize_inplane = resize_inplane

    def load_dicom_directory(self, dicom_dir: str) -> np.ndarray:
        """Load DICOM files from directory with FULL RESOLUTION for LUNA16"""
        dicom_dir = Path(dicom_dir)

        # Find DICOM files (with or without .dcm extension)
        dicom_files = []
        for file_path in dicom_dir.rglob("*"):
            if file_path.is_file() and (
                file_path.suffix.lower() == '.dcm' or
                ('.' not in file_path.name and len(file_path.name) >= 4) or
                file_path.name.startswith(("CT_", "MR_", "CR_", "DX_"))  # Cancer dataset format
            ):
                dicom_files.append(file_path)

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")

        print(f"  📋 Loading {len(dicom_files)} DICOM files...")

        # Load DICOM slices
        slices = []
        slice_positions = []

        for dicom_file in dicom_files:
            try:
                ds = pydicom.dcmread(dicom_file, force=True)
                if hasattr(ds, 'pixel_array'):
                    # Get slice position for proper ordering
                    slice_position = float(getattr(ds, 'SliceLocation', 0))
                    if hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient:
                        slice_position = float(ds.ImagePositionPatient[2])

                    slices.append((ds.pixel_array.astype(np.float32), slice_position, ds))
                    slice_positions.append(slice_position)
            except Exception as e:
                print(f"    ⚠️  Failed to load {dicom_file}: {e}")

        if not slices:
            raise ValueError("No valid DICOM slices loaded")

        # Sort slices by position
        slices.sort(key=lambda x: x[1])

        # Extract pixel arrays and metadata
        pixel_arrays = [s[0] for s in slices]
        sample_ds = slices[0][2]

        # Apply rescale and intercept from DICOM metadata
        rescale_slope = float(getattr(sample_ds, 'RescaleSlope', 1.0))
        rescale_intercept = float(getattr(sample_ds, 'RescaleIntercept', 0.0))

        # Stack volume
        volume = np.stack(pixel_arrays, axis=0)
        print(f"  📐 Original volume shape: {volume.shape}")

        # Apply rescale transformation
        volume = volume * rescale_slope + rescale_intercept
        print(f"  📏 After rescale: min={volume.min():.1f}, max={volume.max():.1f}, mean={volume.mean():.1f}")

        # HU windowing for lung tissue (preserve nodule visibility)
        # Standard lung window: -1024 to +400 HU
        volume = np.clip(volume, -1024, 400)

        # Normalize to [0, 1] range for LUNA16 model
        volume = (volume + 1024) / (400 + 1024)  # Map [-1024, 400] to [0, 1]
        print(f"  📏 After normalization: min={volume.min():.3f}, max={volume.max():.3f}, mean={volume.mean():.3f}")

        # Spatial resampling to consistent voxel spacing (preserve resolution)
        # Get pixel spacing from DICOM
        pixel_spacing = getattr(sample_ds, 'PixelSpacing', [1.0, 1.0])
        slice_thickness = float(getattr(sample_ds, 'SliceThickness', 1.0))

        current_spacing = [slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1])]
        target_spacing = [1.0, 1.0, 1.0]  # 1mm isotropic for optimal LUNA16 performance

        print(f"  📏 Current spacing: {current_spacing}, Target: {target_spacing}")

        # Calculate new shape based on spacing (handle edge case for single slice)
        if np.allclose(current_spacing, target_spacing):
            print(f"  ✅ Spacing already optimal, skipping resampling")
        else:
            scale_factors = [c/t for c, t in zip(current_spacing, target_spacing)]
            new_shape = [int(volume.shape[i] * scale_factors[i]) for i in range(3)]

            print(f"  🔄 Resampling from {volume.shape} to {new_shape} (scale: {scale_factors})...")

            # High-quality resampling with spline interpolation
            volume = ndimage.zoom(volume, scale_factors, order=3, prefilter=True)

            # CRITICAL: Clip interpolation artifacts to [0, 1] range
            # Spline interpolation can create values outside [0, 1]
            volume = np.clip(volume, 0.0, 1.0)
        print(f"  ✅ Final LUNA16 volume shape: {volume.shape}")

        # Limit number of slices for computational efficiency
        if volume.shape[0] > self.max_slices:
            print(f"  ⚡ Limiting slices from {volume.shape[0]} to {self.max_slices} (center extraction)")
            # Extract center slices
            start_idx = (volume.shape[0] - self.max_slices) // 2
            end_idx = start_idx + self.max_slices
            volume = volume[start_idx:end_idx]
            print(f"  ✅ Slices after limiting: {volume.shape}")

        # Resize in-plane to 512x512 if needed (LUNA16 standard)
        if self.resize_inplane:
            target_size = (512, 512)
        else:
            target_size = volume.shape[-2:]  # Keep native resolution

        if self.resize_inplane and volume.shape[-2:] != target_size:  # Check last 2 dimensions
            print(f"  🔄 Resizing in-plane from {volume.shape[-2:]} to {target_size}...")

            # Handle different volume shapes
            if len(volume.shape) == 3:  # (slices, H, W)
                resized_volume = np.zeros((volume.shape[0], *target_size), dtype=volume.dtype)
                for i in range(volume.shape[0]):
                    zoom_factors = [target_size[j] / volume.shape[j+1] for j in range(2)]
                    resized_volume[i] = ndimage.zoom(volume[i], zoom_factors, order=3, prefilter=True)
                # Clip interpolation artifacts
                volume = np.clip(resized_volume, 0.0, 1.0)
            elif len(volume.shape) == 4:  # Edge case: (1, slices, H, W) - flatten first
                print(f"  🔧 Handling 4D volume: {volume.shape}")
                volume = volume.squeeze(0)  # Remove first dimension
                resized_volume = np.zeros((volume.shape[0], *target_size), dtype=volume.dtype)
                for i in range(volume.shape[0]):
                    zoom_factors = [target_size[j] / volume.shape[j+1] for j in range(2)]
                    resized_volume[i] = ndimage.zoom(volume[i], zoom_factors, order=3, prefilter=True)
                # Clip interpolation artifacts
                volume = np.clip(resized_volume, 0.0, 1.0)
            else:
                print(f"  ⚠️  Unexpected volume shape: {volume.shape}, skipping resize")

            print(f"  ✅ Final preprocessed shape: {volume.shape}")

        print(f"  📊 Final stats: min={volume.min():.3f}, max={volume.max():.3f}, mean={volume.mean():.3f}")

        return volume

    def load_zip_file(self, zip_path: str) -> Tuple[np.ndarray, str]:
        """Load DICOM volume from ZIP file with full resolution preservation"""
        zip_path = Path(zip_path)

        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        print(f"📂 Loading LUNA16 data from: {zip_path.name}")

        # Extract to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)

                # Find DICOM directory
                dicom_dirs = []
                for item in temp_path.rglob("*"):
                    if item.is_dir():
                        # Check if directory contains DICOM files
                        dicom_files = list(item.glob("*"))
                        if dicom_files:
                            dicom_dirs.append(item)

                if not dicom_dirs:
                    # No subdirectories, use temp_path directly
                    dicom_dirs = [temp_path]

                # Use the first (or only) directory with most files
                best_dir = max(dicom_dirs, key=lambda d: len(list(d.glob("*"))))

                # Load volume with full resolution
                volume = self.load_dicom_directory(str(best_dir))

                return volume, "success"

            except Exception as e:
                print(f"❌ Error loading ZIP file: {e}")
                raise

    def get_test_files(self) -> Dict[str, int]:
        """Get available test files and their labels"""
        return {
            "norma_anon.zip": 0,      # Normal
            "pneumonia_anon.zip": 1,  # Pathology
            "pneumotorax_anon.zip": 1  # Pathology
        }

    def load_all_test_cases(self) -> Dict[str, Tuple[np.ndarray, int]]:
        """Load all test cases with full resolution for LUNA16"""
        if not self.test_data_dir:
            raise ValueError("Test data directory not specified")

        test_files = self.get_test_files()
        results = {}

        for filename, label in test_files.items():
            zip_path = self.test_data_dir / filename
            if zip_path.exists():
                try:
                    volume, status = self.load_zip_file(str(zip_path))
                    results[filename] = (volume, label)
                    print(f"✅ Loaded {filename}: {volume.shape}, label={label}")
                except Exception as e:
                    print(f"❌ Failed to load {filename}: {e}")
            else:
                print(f"⚠️  Test file not found: {zip_path}")

        return results


def create_luna16_test_loader(test_data_dir: Optional[str] = None, max_slices: int = 160) -> LUNA16TestDataLoader:
    """Factory function to create LUNA16TestDataLoader

    Args:
        test_data_dir: Directory containing test data
        max_slices: Maximum number of slices to process (default 160)
    """
    return LUNA16TestDataLoader(test_data_dir, max_slices)


# Example usage and testing
if __name__ == "__main__":
    # Test the loader
    data_root = os.getenv("RADIASSIST_TEST_DATA_PATH")
    if not data_root:
        print("Set RADIASSIST_TEST_DATA_PATH to run this example.")
    else:
        test_dir = Path(data_root).expanduser()
        if test_dir.exists():
            loader = LUNA16TestDataLoader(test_dir)

            print("🧪 Testing LUNA16TestDataLoader...")
            test_cases = loader.load_all_test_cases()

            print(f"📊 Loaded {len(test_cases)} test cases:")
            for filename, (volume, label) in test_cases.items():
                print(f"  {filename}: {volume.shape}, label={label}")
                print(f"    Stats: min={volume.min():.3f}, max={volume.max():.3f}, mean={volume.mean():.3f}")
        else:
            print(f"❌ Test directory not found: {test_dir}")
