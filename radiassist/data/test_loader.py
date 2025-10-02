"""
Test data loader for final validation datasets
Handles ZIP archives with DICOM files for final evaluation
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
from torch.utils.data import Dataset
from scipy import ndimage

try:
    from .preprocessing import create_preprocessor
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from preprocessing import create_preprocessor


class TestDataLoader:
    """Load and process test datasets from ZIP archives"""

    def __init__(self, test_data_dir: Optional[Path] = None):
        self.test_data_dir = Path(test_data_dir) if test_data_dir else None
        self.preprocessor = create_preprocessor("covid19")  # Use COVID19 preprocessing

        # Expected test files and their labels
        self.test_files = {
            "norma_anon.zip": 0,      # Normal
            "pneumonia_anon.zip": 1,  # Pathology
            "pneumotorax_anon.zip": 1  # Pathology
        }

    def load_dicom_directory(self, dicom_dir: str) -> np.ndarray:
        """Load DICOM files from directory and preprocess for COVID19 model"""
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

        # Sort files by name for consistent ordering
        dicom_files.sort(key=lambda x: x.name)

        print(f"  📋 Loading {len(dicom_files)} DICOM files...")

        # Load DICOM files
        slices = []
        for dicom_file in dicom_files:
            try:
                ds = pydicom.dcmread(dicom_file, force=True)
                slices.append(ds)
            except Exception as e:
                print(f"  ⚠️  Failed to load {dicom_file}: {e}")
                continue

        if not slices:
            raise ValueError(f"No valid DICOM files loaded from {dicom_dir}")

        # Sort slices by position
        try:
            slices.sort(key=lambda x: float(x.SliceLocation))
        except (AttributeError, ValueError):
            # If SliceLocation not available, sort by InstanceNumber or keep original order
            try:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except (AttributeError, ValueError):
                pass  # Keep original order

        # Extract pixel arrays
        pixel_arrays = []
        for slice_ds in slices:
            pixel_array = slice_ds.pixel_array.astype(np.float32)
            # Handle case where pixel_array might be 3D instead of 2D
            if pixel_array.ndim == 3:
                # If it's 3D, take the first slice or reshape appropriately
                if pixel_array.shape[0] == 1:
                    pixel_array = pixel_array[0]  # Remove singleton dimension
                else:
                    # Multiple slices in one DICOM, take middle slice
                    pixel_array = pixel_array[pixel_array.shape[0] // 2]
            pixel_arrays.append(pixel_array)

        # Stack into volume
        volume = np.stack(pixel_arrays, axis=0)  # [Z, H, W]

        print(f"  📐 Volume shape: {volume.shape}")

        # Manual preprocessing for hackathon test data
        print(f"  🔧 Applying manual preprocessing...")

        # Basic DICOM normalization (convert from raw DICOM values)
        # Handle rescale slope and intercept from first DICOM
        rescale_slope = getattr(slices[0], 'RescaleSlope', 1.0)
        rescale_intercept = getattr(slices[0], 'RescaleIntercept', 0.0)

        # Apply rescale
        volume = volume * rescale_slope + rescale_intercept
        print(f"  📏 After rescale: min={volume.min():.1f}, max={volume.max():.1f}, mean={volume.mean():.1f}")

        # Clip to reasonable HU range for chest CT
        volume = np.clip(volume, -1024, 3071)

        # Normalize to [0, 1] range
        volume = (volume + 1024) / (3071 + 1024)
        print(f"  📏 After normalization: min={volume.min():.3f}, max={volume.max():.3f}, mean={volume.mean():.3f}")

        # Resize to expected shape for COVID19 classifier (trained on 64 slices, 256x256)
        target_slices = 64
        target_size = (256, 256)

        # Interpolate to target number of slices
        if volume.shape[0] != target_slices:
            print(f"  🔄 Interpolating from {volume.shape[0]} to {target_slices} slices...")
            z_indices = np.linspace(0, volume.shape[0] - 1, target_slices)
            volume_interp = []
            for z_idx in z_indices:
                z_floor = int(np.floor(z_idx))
                z_ceil = min(z_floor + 1, volume.shape[0] - 1)
                alpha = z_idx - z_floor

                if z_floor == z_ceil:
                    slice_interp = volume[z_floor]
                else:
                    slice_interp = (1 - alpha) * volume[z_floor] + alpha * volume[z_ceil]

                volume_interp.append(slice_interp)

            volume = np.stack(volume_interp, axis=0)

        # Resize spatial dimensions if needed
        if volume.shape[1:] != target_size:
            print(f"  📐 Resizing from {volume.shape[1:]} to {target_size}...")
            volume_resized = []
            for i in range(volume.shape[0]):
                slice_2d = volume[i]  # Shape: (H, W)
                zoom_factors = (target_size[0]/slice_2d.shape[0], target_size[1]/slice_2d.shape[1])
                resized_slice = ndimage.zoom(slice_2d, zoom_factors, order=1)
                volume_resized.append(resized_slice)
            volume_resized = np.stack(volume_resized, axis=0)
        else:
            volume_resized = volume

        # COVID19 model expects (Z, H, W) not multi-channel
        # Return single-channel volume for slice-by-slice processing
        print(f"  ✅ Final processed shape: {volume_resized.shape}")
        print(f"  📊 Final stats: min={volume_resized.min():.3f}, max={volume_resized.max():.3f}, mean={volume_resized.mean():.3f}")

        return volume_resized  # Shape: (64, 256, 256)

    def load_all_studies(self) -> List[Dict]:
        """Load all test studies with metadata"""
        studies = []

        for zip_name, expected_label in self.test_files.items():
            zip_path = self.test_data_dir / zip_name

            if not zip_path.exists():
                print(f"⚠️  Test file not found: {zip_path}")
                continue

            study_data = self._load_study_from_zip(zip_path, expected_label)
            if study_data:
                studies.append(study_data)

        return studies

    def _load_study_from_zip(self, zip_path: Path, expected_label: int) -> Optional[Dict]:
        """Load single study from ZIP archive"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract ZIP
                with zipfile.ZipFile(zip_path, 'r') as zip_file:
                    zip_file.extractall(temp_path)

                # Find DICOM files
                dicom_files = []
                for file_path in temp_path.rglob("*"):
                    if file_path.is_file():
                        try:
                            # Check if it's a DICOM file
                            ds = pydicom.dcmread(file_path, force=True)
                            if hasattr(ds, 'PixelData'):
                                dicom_files.append(file_path)
                        except:
                            continue

                if not dicom_files:
                    print(f"⚠️  No valid DICOM files in {zip_path.name}")
                    return None

                # Sort by Instance Number or slice location
                dicom_files = self._sort_dicom_files(dicom_files)

                # Load volume data
                volume_data = self._load_dicom_volume(dicom_files)
                if volume_data is None:
                    return None

                # Get study metadata
                first_ds = pydicom.dcmread(dicom_files[0], force=True)
                study_info = {
                    "study_id": getattr(first_ds, 'StudyInstanceUID', zip_path.stem),
                    "zip_name": zip_path.name,
                    "expected_label": expected_label,
                    "num_slices": len(dicom_files),
                    "volume_data": volume_data,
                    "pixel_spacing": getattr(first_ds, 'PixelSpacing', [1.0, 1.0]),
                    "slice_thickness": getattr(first_ds, 'SliceThickness', 1.0)
                }

                print(f"✅ Loaded {zip_path.name}: {len(dicom_files)} slices, "
                      f"shape {volume_data.shape}")

                return study_info

        except Exception as e:
            print(f"❌ Failed to load {zip_path.name}: {e}")
            return None

    def _sort_dicom_files(self, dicom_files: List[Path]) -> List[Path]:
        """Sort DICOM files by slice position"""
        def get_sort_key(file_path):
            try:
                ds = pydicom.dcmread(file_path, force=True)

                # Try Instance Number first
                if hasattr(ds, 'InstanceNumber'):
                    return float(ds.InstanceNumber)

                # Try Image Position Patient Z coordinate
                if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
                    return float(ds.ImagePositionPatient[2])

                # Try Slice Location
                if hasattr(ds, 'SliceLocation'):
                    return float(ds.SliceLocation)

                # Fallback to filename
                return float(file_path.stem.split('_')[-1]) if file_path.stem.split('_')[-1].isdigit() else 0

            except:
                return 0

        return sorted(dicom_files, key=get_sort_key)

    def _load_dicom_volume(self, dicom_files: List[Path]) -> Optional[np.ndarray]:
        """Load DICOM files into 3D volume"""
        try:
            slices = []

            for file_path in dicom_files:
                ds = pydicom.dcmread(file_path, force=True)

                if not hasattr(ds, 'PixelData'):
                    continue

                # Get pixel array
                slice_data = ds.pixel_array.astype(np.float32)

                # Apply rescale slope and intercept if available
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    slice_data = slice_data * ds.RescaleSlope + ds.RescaleIntercept

                slices.append(slice_data)

            if not slices:
                return None

            # Stack into 3D volume (Z, Y, X)
            volume = np.stack(slices, axis=0)

            return volume

        except Exception as e:
            print(f"❌ Error loading DICOM volume: {e}")
            return None


class TestDataset(Dataset):
    """PyTorch Dataset for test data"""

    def __init__(self, test_data_dir: Path):
        self.loader = TestDataLoader(test_data_dir)
        self.studies = self.loader.load_all_studies()

        if not self.studies:
            raise ValueError("No valid test studies found")

        print(f"✅ Loaded {len(self.studies)} test studies")

    def _preprocess_numpy_volume(self, volume: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
        """Basic preprocessing for test volumes"""
        # Handle edge case where volume has wrong dimensions
        if volume.ndim == 4:
            # Remove extra dimension (likely from single slice with extra dim)
            volume = np.squeeze(volume)

        # Ensure 3D volume
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]  # Add Z dimension

        # Apply HU windowing (lung window)
        volume = np.clip(volume, -1000, 400)

        # Normalize to [0, 1]
        volume = (volume + 1000) / 1400.0

        # Resize to standard size if needed
        target_shape = (64, 256, 256)  # Z, Y, X

        if volume.shape != target_shape:
            # Handle different number of dimensions
            if len(volume.shape) != 3:
                print(f"⚠️  Unexpected volume shape: {volume.shape}")
                return np.zeros(target_shape, dtype=np.float32)

            # Calculate zoom factors
            zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
            volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')

        return volume

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> Dict:
        study = self.studies[idx]

        # Preprocess volume
        try:
            volume = study["volume_data"]
            pixel_spacing = study["pixel_spacing"]
            slice_thickness = study["slice_thickness"]

            # Create spacing tuple (slice_thickness, pixel_spacing[0], pixel_spacing[1])
            spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])

            # Apply basic preprocessing directly on numpy array
            processed_volume = self._preprocess_numpy_volume(volume, spacing)

            return {
                "volume": torch.FloatTensor(processed_volume).unsqueeze(0),  # Add channel dim
                "study_id": study["study_id"],
                "expected_label": study["expected_label"],
                "zip_name": study["zip_name"],
                "original_shape": volume.shape
            }

        except Exception as e:
            print(f"❌ Error preprocessing study {study['study_id']}: {e}")
            # Return dummy data to avoid crash
            return {
                "volume": torch.zeros((1, 64, 256, 256)),
                "study_id": study["study_id"],
                "expected_label": study["expected_label"],
                "zip_name": study["zip_name"],
                "original_shape": (1, 1, 1)
            }


def create_test_dataset(test_data_dir: Optional[str] = None) -> TestDataset:
    """Create test dataset for final validation"""
    if test_data_dir is None:
        data_env = os.getenv("RADIASSIST_TEST_DATA_PATH")
        base_dir = Path(data_env).expanduser() if data_env else Path(__file__).resolve().parents[2] / "datasets" / "LCT-dataset"
    else:
        base_dir = Path(test_data_dir).expanduser()
    return TestDataset(base_dir)


if __name__ == "__main__":
    # Quick test
    try:
        dataset = create_test_dataset()
    except Exception as exc:
        print(f"❌ Unable to load test dataset: {exc}")
        print("Set RADIASSIST_TEST_DATA_PATH to point to a directory with hackathon ZIP files.")
    else:
        print("\n🧪 TEST DATASET SUMMARY")
        print("=" * 40)

        for i, sample in enumerate(dataset):
            print(f"Study {i+1}: {sample['zip_name']}")
            print(f"  Expected label: {sample['expected_label']}")
            print(f"  Volume shape: {sample['volume'].shape}")
            print(f"  Original shape: {sample['original_shape']}")
