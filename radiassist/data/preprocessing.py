"""
Adaptive preprocessing pipeline for RadiAssist
Critical component implementing cubic interpolation for COVID19 sparse Z-axis
"""

import logging
import os
import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, Dict, Union, List
from dataclasses import dataclass
from scipy import ndimage
import time

@dataclass
class VolumeMetadata:
    """Metadata for medical volume"""
    original_shape: Tuple[int, int, int]
    original_spacing: Tuple[float, float, float]
    processed_shape: Tuple[int, int, int]
    processed_spacing: Tuple[float, float, float]
    interpolation_method: str
    processing_time: float
    quality_flags: List[str]

class AdaptivePreprocessor:
    """
    Adaptive preprocessing pipeline for multi-modal medical imaging
    Handles COVID19 (sparse Z), LUNA16 (MetaImage), MosMed (DICOM)
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0),
        target_size: Optional[Tuple[int, int, int]] = None,
        hu_window: Tuple[float, float] = (-1000, 400),
        interpolation_method: str = "cubic",
        quality_threshold: float = 0.8
    ):
        """
        Initialize adaptive preprocessor

        Args:
            target_spacing: Target voxel spacing in mm (x, y, z)
            target_size: Target volume size (optional)
            hu_window: HU window for normalization (lung window)
            interpolation_method: Default interpolation method
            quality_threshold: Minimum quality score for processing
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.hu_window = hu_window
        self.interpolation_method = interpolation_method
        self.quality_threshold = quality_threshold

        self.logger = self._setup_logging()

        # Interpolation method mapping
        self.interpolation_orders = {
            "linear": 1,
            "cubic": 3,
            "nearest": 0
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup preprocessing logger"""
        logger = logging.getLogger("AdaptivePreprocessor")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def process_volume(
        self,
        volume_path: Path,
        dataset_type: str = "auto"
    ) -> Tuple[np.ndarray, VolumeMetadata]:
        """
        Main processing entry point for any volume type

        Args:
            volume_path: Path to volume file
            dataset_type: Type of dataset (covid19, luna16, mosmed, auto)

        Returns:
            Processed volume and metadata
        """
        start_time = time.time()

        try:
            # Auto-detect dataset type if needed
            if dataset_type == "auto":
                dataset_type = self._detect_dataset_type(volume_path)

            # Load volume based on type
            if dataset_type == "covid19":
                volume, spacing, metadata = self._load_nifti(volume_path)
            elif dataset_type == "luna16":
                volume, spacing, metadata = self._load_metaimage(volume_path)
            elif dataset_type == "mosmed":
                volume, spacing, metadata = self._load_dicom_series(volume_path)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            # Quality assessment
            quality_score, quality_flags = self._assess_quality(volume, spacing)

            if quality_score < self.quality_threshold:
                self.logger.warning(
                    f"Low quality volume: {quality_score:.3f} < {self.quality_threshold}"
                )

            # Adaptive preprocessing based on spacing
            processed_volume = self._adaptive_preprocess(
                volume, spacing, dataset_type, quality_flags
            )

            processing_time = time.time() - start_time

            # Create metadata
            result_metadata = VolumeMetadata(
                original_shape=volume.shape,
                original_spacing=spacing,
                processed_shape=processed_volume.shape,
                processed_spacing=self.target_spacing,
                interpolation_method=self.interpolation_method,
                processing_time=processing_time,
                quality_flags=quality_flags
            )

            self.logger.info(
                f"Processed {volume_path.name}: "
                f"{volume.shape} → {processed_volume.shape} "
                f"in {processing_time:.2f}s"
            )

            return processed_volume, result_metadata

        except Exception as e:
            self.logger.error(f"Failed to process {volume_path}: {str(e)}")
            raise

    def _detect_dataset_type(self, volume_path: Path) -> str:
        """Auto-detect dataset type from file path/name"""
        if volume_path.suffix in ['.nii', '.nii.gz']:
            return "covid19"
        elif volume_path.suffix == '.mhd':
            return "luna16"
        elif volume_path.suffix in ['.dcm', '.DCM'] or 'dicom' in str(volume_path).lower():
            return "mosmed"
        else:
            raise ValueError(f"Cannot detect dataset type for: {volume_path}")

    def _load_nifti(self, nifti_path: Path) -> Tuple[np.ndarray, Tuple[float, float, float], Dict]:
        """Load NIfTI file (COVID19 dataset)"""
        try:
            img = nib.load(nifti_path)
            volume = img.get_fdata().astype(np.float32)
            spacing = img.header.get_zooms()[:3]

            metadata = {
                "format": "nifti",
                "header": img.header,
                "affine": img.affine
            }

            return volume, spacing, metadata

        except Exception as e:
            raise RuntimeError(f"Failed to load NIfTI {nifti_path}: {str(e)}")

    def _load_metaimage(self, mhd_path: Path) -> Tuple[np.ndarray, Tuple[float, float, float], Dict]:
        """Load MetaImage file (LUNA16 dataset)"""
        try:
            # Use SimpleITK for MetaImage
            img = sitk.ReadImage(str(mhd_path))
            volume = sitk.GetArrayFromImage(img).astype(np.float32)

            # SimpleITK returns spacing in (x, y, z) order
            spacing = img.GetSpacing()

            # Volume from SimpleITK is in (z, y, x) order, transpose to (x, y, z)
            volume = np.transpose(volume, (2, 1, 0))

            metadata = {
                "format": "metaimage",
                "spacing": spacing,
                "origin": img.GetOrigin(),
                "direction": img.GetDirection()
            }

            return volume, spacing, metadata

        except Exception as e:
            raise RuntimeError(f"Failed to load MetaImage {mhd_path}: {str(e)}")

    def _load_dicom_series(self, dicom_path: Path) -> Tuple[np.ndarray, Tuple[float, float, float], Dict]:
        """Load DICOM series (MosMed dataset)"""
        try:
            if dicom_path.is_file():
                # Single DICOM file
                ds = pydicom.dcmread(dicom_path)
                volume = ds.pixel_array.astype(np.float32)

                # Apply rescale slope and intercept
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    volume = volume * ds.RescaleSlope + ds.RescaleIntercept

                # Get spacing
                pixel_spacing = ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else [1.0, 1.0]
                slice_thickness = ds.SliceThickness if hasattr(ds, 'SliceThickness') else 1.0
                spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness))

                # Add channel dimension for single slice
                if volume.ndim == 2:
                    volume = volume[:, :, np.newaxis]

            else:
                # DICOM series directory
                raise NotImplementedError("DICOM series loading not yet implemented")

            metadata = {
                "format": "dicom",
                "study_uid": ds.StudyInstanceUID if hasattr(ds, 'StudyInstanceUID') else None,
                "series_uid": ds.SeriesInstanceUID if hasattr(ds, 'SeriesInstanceUID') else None
            }

            return volume, spacing, metadata

        except Exception as e:
            raise RuntimeError(f"Failed to load DICOM {dicom_path}: {str(e)}")

    def _assess_quality(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float]
    ) -> Tuple[float, List[str]]:
        """Assess volume quality and return score + flags"""
        quality_flags = []
        quality_score = 1.0

        # Check for sparse Z-axis (COVID19 issue)
        if spacing[2] > 5.0:
            quality_flags.append("SPARSE_Z_AXIS")
            quality_score *= 0.8

        # Check for very anisotropic voxels
        max_spacing = max(spacing)
        min_spacing = min(spacing)
        if max_spacing / min_spacing > 5:
            quality_flags.append("ANISOTROPIC_VOXELS")
            quality_score *= 0.9

        # Check for unusual volume dimensions
        if volume.shape[2] < 10:
            quality_flags.append("FEW_SLICES")
            quality_score *= 0.7

        # Check for reasonable HU range (if CT data)
        hu_min, hu_max = volume.min(), volume.max()
        if hu_min < -2000 or hu_max > 5000:
            quality_flags.append("UNUSUAL_HU_RANGE")
            quality_score *= 0.9

        return quality_score, quality_flags

    def _adaptive_preprocess(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        dataset_type: str,
        quality_flags: List[str]
    ) -> np.ndarray:
        """Adaptive preprocessing based on volume characteristics"""

        processed_volume = volume.copy()

        # Step 1: HU windowing and normalization
        processed_volume = self._normalize_hu(processed_volume)

        # Step 2: Adaptive resampling based on spacing
        if self._needs_resampling(original_spacing, quality_flags):
            processed_volume = self._resample_volume(
                processed_volume, original_spacing, self.target_spacing
            )

        # Step 3: Dataset-specific processing
        if dataset_type == "covid19":
            processed_volume = self._covid19_specific_preprocessing(processed_volume)
        elif dataset_type == "luna16":
            processed_volume = self._luna16_specific_preprocessing(processed_volume)
        elif dataset_type == "mosmed":
            processed_volume = self._mosmed_specific_preprocessing(processed_volume)

        return processed_volume

    def _normalize_hu(self, volume: np.ndarray) -> np.ndarray:
        """Normalize HU values to [0, 1] range using lung window"""
        # Apply lung window
        windowed = np.clip(volume, self.hu_window[0], self.hu_window[1])

        # Normalize to [0, 1]
        hu_range = self.hu_window[1] - self.hu_window[0]
        normalized = (windowed - self.hu_window[0]) / hu_range

        return normalized.astype(np.float32)

    def _needs_resampling(
        self,
        original_spacing: Tuple[float, float, float],
        quality_flags: List[str]
    ) -> bool:
        """Determine if resampling is needed"""
        # Always resample if sparse Z-axis
        if "SPARSE_Z_AXIS" in quality_flags:
            return True

        # Resample if spacing differs significantly from target
        spacing_ratio = np.array(original_spacing) / np.array(self.target_spacing)
        if np.any(spacing_ratio > 1.5) or np.any(spacing_ratio < 0.7):
            return True

        return False

    def _resample_volume(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Resample volume to target spacing using cubic interpolation
        Critical component for COVID19 sparse Z-axis resolution
        """
        # Calculate zoom factors
        zoom_factors = np.array(original_spacing) / np.array(target_spacing)

        self.logger.info(
            f"Resampling: {original_spacing} → {target_spacing}, "
            f"zoom factors: {zoom_factors}"
        )

        # Use cubic interpolation (order=3) based on test results
        interpolation_order = self.interpolation_orders[self.interpolation_method]

        resampled = ndimage.zoom(
            volume,
            zoom_factors,
            order=interpolation_order,
            mode='nearest',
            prefilter=True
        )

        self.logger.info(
            f"Resampling completed: {volume.shape} → {resampled.shape}"
        )

        return resampled

    def _covid19_specific_preprocessing(self, volume: np.ndarray) -> np.ndarray:
        """COVID19-specific preprocessing steps"""
        # Specific preprocessing for inflammation detection
        # Could include noise reduction, contrast enhancement, etc.
        return volume

    def _luna16_specific_preprocessing(self, volume: np.ndarray) -> np.ndarray:
        """LUNA16-specific preprocessing steps"""
        # Specific preprocessing for nodule detection
        # Could include noise reduction optimized for nodules
        return volume

    def _mosmed_specific_preprocessing(self, volume: np.ndarray) -> np.ndarray:
        """MosMed-specific preprocessing steps"""
        # Specific preprocessing for cancer classification
        # Could include contrast optimization, etc.
        return volume

def create_preprocessor(dataset_type: str = "covid19") -> AdaptivePreprocessor:
    """Factory function to create dataset-specific preprocessor"""

    if dataset_type == "covid19":
        # Optimized for COVID19 sparse Z-axis interpolation
        return AdaptivePreprocessor(
            target_spacing=(1.0, 1.0, 2.0),  # 2mm Z-spacing from test results
            interpolation_method="cubic",     # Winner from interpolation tests
            hu_window=(-1000, 400)           # Lung window
        )

    elif dataset_type == "luna16":
        # Optimized for LUNA16 nodule detection
        return AdaptivePreprocessor(
            target_spacing=(1.0, 1.0, 1.0),  # Isotropic 1mm spacing
            interpolation_method="cubic",
            hu_window=(-1000, 400)
        )

    elif dataset_type == "mosmed":
        # Optimized for MosMed cancer classification
        return AdaptivePreprocessor(
            target_spacing=(1.0, 1.0, 2.0),  # Match COVID19 for consistency
            interpolation_method="cubic",
            hu_window=(-1000, 400)
        )

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

# Example usage
if __name__ == "__main__":
    # Test with COVID19 data
    preprocessor = create_preprocessor("covid19")

    sample_volume = os.getenv("RADIASSIST_SAMPLE_VOLUME")
    if not sample_volume:
        print("Set RADIASSIST_SAMPLE_VOLUME to run this example.")
    else:
        volume_path = Path(sample_volume).expanduser()
        if volume_path.exists():
            try:
                processed_volume, metadata = preprocessor.process_volume(volume_path, "covid19")
                print(f"Successfully processed: {metadata}")
            except Exception as e:
                print(f"Processing failed: {e}")
        else:
            print(f"Example volume not found: {volume_path}")
