"""
DICOM Loader for COVID19 Classifier Inference
Converts DICOM studies to preprocessed volumes for model inference
"""

import pydicom
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging
from scipy import ndimage
import zipfile
import tempfile
import shutil

logger = logging.getLogger(__name__)


@dataclass
class StudyMetadata:
    """Metadata extracted from DICOM study"""
    study_uid: str
    series_uid: str
    slice_count: int
    spacing: Tuple[float, float, float]
    shape: Tuple[int, int, int]
    modality: str
    body_part: str
    patient_id: Optional[str] = None


class DICOMLoader:
    """
    Loads DICOM studies and applies COVID19 preprocessing pipeline
    Compatible with NIfTI preprocessing: (64, 256, 256) output
    """

    def __init__(
        self,
        target_depth: int = 64,
        target_shape: Tuple[int, int] = (256, 256),
        target_spacing_xy: float = 2.0,
        target_spacing_z: float = 2.0,
        hu_window: Tuple[float, float] = (-1000, 200),
        lung_threshold: float = -300.0
    ):
        """
        Initialize DICOM loader with COVID19-compatible preprocessing

        Args:
            target_depth: Number of slices to select (64 for COVID19 model)
            target_shape: Target in-plane resolution (256x256)
            target_spacing_xy: Target in-plane spacing in mm
            target_spacing_z: Target slice spacing in mm (2.0mm for interpolation)
            hu_window: HU window for lung visualization
            lung_threshold: HU threshold for lung ROI extraction
        """
        self.target_depth = target_depth
        self.target_shape = target_shape
        self.target_spacing_xy = target_spacing_xy
        self.target_spacing_z = target_spacing_z
        self.hu_window = hu_window
        self.lung_threshold = lung_threshold

    def load_dicom_directory(
        self,
        dicom_dir: Path
    ) -> Tuple[np.ndarray, StudyMetadata]:
        """
        Load DICOM directory and preprocess to (64, 256, 256)

        Args:
            dicom_dir: Path to directory containing DICOM files

        Returns:
            Preprocessed volume (D, H, W) and metadata
        """
        # Find all DICOM files
        dicom_files = self._find_dicom_files(dicom_dir)
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")

        logger.info(f"Found {len(dicom_files)} DICOM files in {dicom_dir}")

        # Load volume using SimpleITK
        volume, metadata = self._load_volume_sitk(dicom_files, dicom_dir)

        # Apply COVID19 preprocessing pipeline
        processed_volume = self._preprocess_volume(volume, metadata.spacing)

        return processed_volume, metadata

    def load_dicom_zip(
        self,
        zip_path: Path,
        extract_dir: Optional[Path] = None
    ) -> Tuple[np.ndarray, StudyMetadata]:
        """
        Load DICOM study from ZIP archive

        Args:
            zip_path: Path to ZIP file containing DICOM study
            extract_dir: Optional directory for extraction (temp dir if None)

        Returns:
            Preprocessed volume and metadata
        """
        # Create temp directory if not provided
        cleanup_temp = False
        if extract_dir is None:
            extract_dir = Path(tempfile.mkdtemp(prefix="dicom_"))
            cleanup_temp = True

        try:
            # Extract ZIP
            logger.info(f"Extracting {zip_path} to {extract_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find the actual DICOM directory (may be nested)
            dicom_dir = self._find_dicom_directory(extract_dir)

            # Load and preprocess
            volume, metadata = self.load_dicom_directory(dicom_dir)

            return volume, metadata

        finally:
            # Cleanup temp directory if created
            if cleanup_temp and extract_dir.exists():
                shutil.rmtree(extract_dir)

    def _find_dicom_files(self, directory: Path) -> List[Path]:
        """Find all DICOM files in directory"""
        dicom_files = []

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                # Check by extension or try to read as DICOM
                if file_path.suffix.lower() in ['.dcm', '.dicom']:
                    dicom_files.append(file_path)
                elif '.' not in file_path.name:
                    # Files without extension might be DICOM
                    try:
                        pydicom.dcmread(file_path, stop_before_pixels=True)
                        dicom_files.append(file_path)
                    except:
                        continue

        return sorted(dicom_files)

    def _find_dicom_directory(self, root_dir: Path) -> Path:
        """Find directory containing DICOM files (handles nested structure)"""
        # Try root directory first
        if self._find_dicom_files(root_dir):
            return root_dir

        # Search subdirectories
        for subdir in root_dir.rglob("*"):
            if subdir.is_dir() and self._find_dicom_files(subdir):
                return subdir

        raise ValueError(f"No DICOM files found in {root_dir} or subdirectories")

    def _load_volume_sitk(
        self,
        dicom_files: List[Path],
        dicom_dir: Path
    ) -> Tuple[np.ndarray, StudyMetadata]:
        """
        Load DICOM series using SimpleITK

        Returns:
            Volume array (Z, Y, X) and metadata
        """
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))

        if not dicom_names:
            # Fallback: use found files directly
            dicom_names = [str(f) for f in dicom_files]

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Convert to numpy (Z, Y, X)
        volume = sitk.GetArrayFromImage(image)

        # Get metadata
        spacing = image.GetSpacing()  # (X, Y, Z)
        spacing_zyx = (spacing[2], spacing[1], spacing[0])  # Convert to (Z, Y, X)

        # Extract DICOM metadata
        sample_ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)

        study_uid = getattr(sample_ds, 'StudyInstanceUID', 'UNKNOWN')
        series_uid = getattr(sample_ds, 'SeriesInstanceUID', 'UNKNOWN')
        modality = getattr(sample_ds, 'Modality', 'UNKNOWN')
        body_part = getattr(sample_ds, 'BodyPartExamined', 'UNKNOWN')
        patient_id = getattr(sample_ds, 'PatientID', None)

        metadata = StudyMetadata(
            study_uid=study_uid,
            series_uid=series_uid,
            slice_count=volume.shape[0],
            spacing=spacing_zyx,
            shape=volume.shape,
            modality=modality,
            body_part=body_part,
            patient_id=patient_id
        )

        logger.info(f"Loaded volume: shape={volume.shape}, spacing={spacing_zyx}")

        return volume, metadata

    def _preprocess_volume(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Apply COVID19-compatible preprocessing pipeline

        Pipeline:
        1. HU clipping and normalization
        2. Z-axis cubic interpolation (original spacing → 2mm)
        3. XY resampling to 2mm
        4. Lung ROI extraction
        5. Slice selection (64 slices with max lung area)
        6. Final resize to (64, 256, 256)

        Args:
            volume: Input volume (Z, Y, X)
            original_spacing: Original spacing (Z, Y, X) in mm

        Returns:
            Processed volume (64, 256, 256)
        """
        # Step 1: HU clipping and normalization
        volume = self._apply_hu_window(volume)

        # Step 2: Cubic interpolation on Z-axis
        volume = self._resample_z_axis(volume, original_spacing)

        # Step 3: Resample XY to target spacing
        volume = self._resample_xy(volume, original_spacing)

        # Step 4: Extract lung ROI
        lung_mask = self._extract_lung_roi(volume)
        volume = volume * lung_mask  # Mask non-lung regions

        # Step 5: Select 64 slices with maximum lung area
        volume = self._select_slices(volume, lung_mask)

        # Step 6: Resize to target shape
        volume = self._resize_to_target(volume)

        return volume

    def _apply_hu_window(self, volume: np.ndarray) -> np.ndarray:
        """Apply HU window and normalize to [0, 1]"""
        # Robust clipping using percentiles
        p1, p99 = np.percentile(volume, [1, 99])

        # Use percentile or fallback to lung window
        if -2000 <= p1 <= -500 and -200 <= p99 <= 3000:
            hu_min, hu_max = p1, p99
        else:
            hu_min, hu_max = self.hu_window

        # Clip and normalize
        volume = np.clip(volume, hu_min, hu_max)
        volume = (volume - hu_min) / (hu_max - hu_min)

        return volume.astype(np.float32)

    def _resample_z_axis(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Resample Z-axis using cubic interpolation
        Critical for COVID19 8mm → 2mm interpolation
        """
        z_spacing_orig = original_spacing[0]
        z_spacing_target = self.target_spacing_z

        # Calculate zoom factor for Z-axis
        zoom_factor_z = z_spacing_orig / z_spacing_target

        # Only interpolate if spacing is significantly different
        if abs(zoom_factor_z - 1.0) > 0.1:
            logger.info(f"Z-axis interpolation: {z_spacing_orig:.2f}mm → {z_spacing_target:.2f}mm (zoom={zoom_factor_z:.2f})")

            # Cubic interpolation on Z-axis only
            zoom_factors = [zoom_factor_z, 1.0, 1.0]  # (Z, Y, X)
            volume = ndimage.zoom(volume, zoom_factors, order=3, mode='nearest')

        return volume

    def _resample_xy(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float]
    ) -> np.ndarray:
        """Resample XY plane to target spacing"""
        y_spacing_orig = original_spacing[1]
        x_spacing_orig = original_spacing[2]

        zoom_y = y_spacing_orig / self.target_spacing_xy
        zoom_x = x_spacing_orig / self.target_spacing_xy

        # Only resample if significantly different
        if abs(zoom_y - 1.0) > 0.1 or abs(zoom_x - 1.0) > 0.1:
            logger.info(f"XY resampling: ({y_spacing_orig:.2f}, {x_spacing_orig:.2f})mm → {self.target_spacing_xy:.2f}mm")

            zoom_factors = [1.0, zoom_y, zoom_x]  # (Z, Y, X)
            volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')

        return volume

    def _extract_lung_roi(self, volume: np.ndarray) -> np.ndarray:
        """
        Extract lung ROI using thresholding and morphology
        Compatible with COVID19 preprocessing
        """
        # Threshold (HU < -300 in normalized space)
        # After normalization: HU=-300 → normalized value
        hu_min, hu_max = self.hu_window
        threshold_normalized = (self.lung_threshold - hu_min) / (hu_max - hu_min)

        lung_mask = volume < threshold_normalized

        # Morphological operations
        struct = ndimage.generate_binary_structure(3, 2)

        # Opening: remove small noise
        lung_mask = ndimage.binary_opening(lung_mask, structure=struct, iterations=2)

        # Closing: fill small holes
        lung_mask = ndimage.binary_closing(lung_mask, structure=struct, iterations=2)

        # Fill holes slice by slice
        for i in range(lung_mask.shape[0]):
            lung_mask[i] = ndimage.binary_fill_holes(lung_mask[i])

        return lung_mask.astype(np.float32)

    def _select_slices(
        self,
        volume: np.ndarray,
        lung_mask: np.ndarray
    ) -> np.ndarray:
        """
        Select 64 slices with maximum lung area
        Uses Gaussian weighting to prefer central slices
        """
        # Calculate lung area per slice
        lung_areas = lung_mask.sum(axis=(1, 2))

        # Gaussian weighting (prefer center)
        z_center = len(lung_areas) / 2
        z_positions = np.arange(len(lung_areas))
        gaussian_weights = np.exp(-((z_positions - z_center) ** 2) / (2 * (len(lung_areas) / 4) ** 2))

        # Weighted scores
        scores = lung_areas * gaussian_weights

        # Sort and select top 64 slices
        if len(scores) <= self.target_depth:
            # Pad if too few slices
            selected_indices = np.arange(len(scores))
            padding = self.target_depth - len(scores)
            selected_volume = volume[selected_indices]

            if padding > 0:
                pad_width = ((0, padding), (0, 0), (0, 0))
                selected_volume = np.pad(selected_volume, pad_width, mode='edge')
        else:
            # Select top slices and sort by position
            top_indices = np.argsort(scores)[-self.target_depth:]
            selected_indices = np.sort(top_indices)
            selected_volume = volume[selected_indices]

        return selected_volume

    def _resize_to_target(self, volume: np.ndarray) -> np.ndarray:
        """Resize volume to target shape (64, 256, 256)"""
        current_shape = volume.shape

        if current_shape == (self.target_depth, *self.target_shape):
            return volume

        # Calculate zoom factors
        zoom_factors = [
            self.target_depth / current_shape[0],
            self.target_shape[0] / current_shape[1],
            self.target_shape[1] / current_shape[2]
        ]

        # Resize using linear interpolation
        volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')

        return volume


def load_dicom_for_inference(
    path: Path,
    loader: Optional[DICOMLoader] = None
) -> Tuple[np.ndarray, StudyMetadata]:
    """
    Convenience function to load DICOM study (directory or ZIP)

    Args:
        path: Path to DICOM directory or ZIP file
        loader: Optional DICOMLoader instance (creates default if None)

    Returns:
        Preprocessed volume (64, 256, 256) and metadata
    """
    if loader is None:
        loader = DICOMLoader()

    if path.is_dir():
        return loader.load_dicom_directory(path)
    elif path.suffix.lower() == '.zip':
        return loader.load_dicom_zip(path)
    else:
        raise ValueError(f"Unsupported path type: {path}")