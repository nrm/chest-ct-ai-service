"""
Chest CT multi-organ segmentation tool.

This script loads chest CT data (DICOM series or NIfTI volumes) and produces a
multi-label segmentation covering lungs, heart, bones, and remaining soft
tissues. The implementation relies on heuristic intensity thresholds and simple
morphological processing so it can run without a trained neural network.
"""

from __future__ import annotations

import glob
import json
import time
import os
import traceback
import shutil
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # render previews without requiring a GUI backend
from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt
import numpy as np

# Импорт модуля для анализа патологий
try:
    from ct_pathology_analysis import PathologyAnalyzer
    PATHOLOGY_ANALYSIS_AVAILABLE = True
except ImportError:
    PATHOLOGY_ANALYSIS_AVAILABLE = False
    print("Предупреждение: Модуль анализа патологий недоступен")
import SimpleITK as sitk
import scipy.ndimage as ndi
from scipy.ndimage import label
from skimage import measure, morphology
from skimage.segmentation import clear_border

try:
    from openpyxl import Workbook
except Exception:  # pragma: no cover - optional dependency
    Workbook = None

try:
    import py7zr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    py7zr = None

try:
    import rarfile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rarfile = None

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

LABELS = {
    0: "background",
    1: "lungs",
    2: "heart",
    3: "bones",
    4: "other_soft_tissues",
}

TARGET_SEGMENTATION_SPACING_MM = (1.8, 1.8, 1.8)  # (z, y, x) target resolution for processing
TARGET_SEGMENTATION_SPACING_XYZ = tuple(float(v) for v in reversed(TARGET_SEGMENTATION_SPACING_MM))

TARGET_SPACING_SCALE = 3.0
TARGET_SPACING_CAP_MM = 1.2


def compute_target_spacing_xyz(original_spacing_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    target = []
    for os in original_spacing_xyz:
        scaled = min(TARGET_SPACING_CAP_MM, os * TARGET_SPACING_SCALE)
        target.append(max(os, scaled))
    return tuple(target)



def is_archive(path: str) -> bool:
    """Return True if the path looks like an archive that needs unpacking."""
    lower = path.lower()
    if lower.endswith((".nii", ".nii.gz")):
        return False
    if lower.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        return True
    ext = os.path.splitext(lower)[1]
    return ext in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}


def unpack_archive_any(src_path: str, dst_dir: str) -> None:
    """Extract *src_path* into *dst_dir* supporting common archive formats."""
    lower = src_path.lower()
    if lower.endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".tar")):
        with tarfile.open(src_path) as tf:
            tf.extractall(dst_dir)
        return
    if lower.endswith(".zip"):
        with zipfile.ZipFile(src_path) as zf:
            zf.extractall(dst_dir)
        return
    if lower.endswith(".7z"):
        if not py7zr:
            raise RuntimeError(".7z archive requires py7zr package")
        with py7zr.SevenZipFile(src_path) as zf:
            zf.extractall(path=dst_dir)
        return
    if lower.endswith(".rar"):
        if not rarfile:
            raise RuntimeError(".rar archive requires rarfile package")
        with rarfile.RarFile(src_path) as rf:
            rf.extractall(dst_dir)
        return
    shutil.unpack_archive(src_path, dst_dir)


def is_dicom_file(path: str) -> bool:
    """Heuristically check whether the file is a DICOM slice."""
    try:
        with open(path, "rb") as fb:
            header = fb.read(132)
    except OSError:
        return False
    return len(header) >= 132 #and header[128:132] == b"DICM"


def looks_like_dicom_dir(directory: str) -> bool:
    """Return True if the directory appears to contain DICOM files."""
    count = 0
    for root, _, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(".dcm"):
                count += 1
            else:
                if is_dicom_file(os.path.join(root, name)):
                    count += 1
            if count > 10:
                return True
    return count > 0


def collect_dicom_files(directory: str) -> List[str]:
    """Collect DICOM file paths in a directory tree, sorted lexicographically."""
    files: List[str] = []
    for root, _, filenames in os.walk(directory):
        for name in sorted(filenames):
            full = os.path.join(root, name)
            if os.path.isfile(full) and is_dicom_file(full):
                files.append(full)
    return files


def _parse_float_list(value: str) -> List[float]:
    parts = [seg.strip() for seg in value.replace(',', '\\').split('\\') if seg.strip()]
    return [float(seg) for seg in parts]


def _order_dicom_series(file_paths: List[str]) -> List[str]:
    entries: List[Dict[str, object]] = []
    reader = sitk.ImageFileReader()
    for idx, path in enumerate(file_paths):
        try:
            reader.SetFileName(path)
            reader.ReadImageInformation()
        except Exception:
            entries.append({
                "path": path,
                "series_uid": "__unknown__",
                "position": None,
                "instance": None,
                "acq": "",
                "index": idx,
            })
            continue

        def get_tag(tag: str) -> str | None:
            if reader.HasMetaDataKey(tag):
                value = reader.GetMetaData(tag)
                return value.strip() if value is not None else None
            return None

        series_uid = get_tag("0020|000e") or "__unknown__"
        inst_raw = get_tag("0020|0013")
        instance_number = None
        if inst_raw:
            try:
                instance_number = int(round(float(inst_raw)))
            except ValueError:
                instance_number = None

        position = None
        ipp_raw = get_tag("0020|0032")
        iop_raw = get_tag("0020|0037")
        slice_loc_raw = get_tag("0020|1041")
        if ipp_raw and iop_raw:
            try:
                ipp = _parse_float_list(ipp_raw)
                iop = _parse_float_list(iop_raw)
                if len(ipp) >= 3 and len(iop) >= 6:
                    row = np.array(iop[:3])
                    col = np.array(iop[3:6])
                    normal = np.cross(row, col)
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal /= norm
                        position = float(np.dot(normal, np.array(ipp[:3])))
            except Exception:
                position = None
        if position is None and slice_loc_raw:
            try:
                position = float(slice_loc_raw)
            except ValueError:
                position = None

        acq_time = get_tag("0008|0032") or ""

        entries.append({
            "path": path,
            "series_uid": series_uid,
            "position": position,
            "instance": instance_number,
            "acq": acq_time,
            "index": idx,
        })

    if not entries:
        return sorted(file_paths)

    series_groups: Dict[str, List[Dict[str, object]]] = {}
    for entry in entries:
        key = entry["series_uid"]
        series_groups.setdefault(key, []).append(entry)

    def series_rank(item: tuple[str, List[Dict[str, object]]]) -> tuple[int, int]:
        series_entries = item[1]
        return (len(series_entries), -min(int(e["index"]) for e in series_entries))

    best_uid, best_entries = max(series_groups.items(), key=series_rank)

    def sort_key(entry: Dict[str, object]) -> tuple[int, float, int, int, str, int]:
        pos = entry["position"]
        inst = entry["instance"]
        idx_val = int(entry["index"])
        acq = entry["acq"] if isinstance(entry["acq"], str) else ""
        pos_flag = 0 if pos is not None else 1
        pos_val = float(pos) if pos is not None else 0.0
        inst_flag = 0 if inst is not None else 1
        inst_val = int(inst) if inst is not None else 1_000_000 + idx_val
        return (pos_flag, pos_val, inst_flag, inst_val, acq, idx_val)

    sorted_entries = sorted(best_entries, key=sort_key)

    positions = [e["position"] for e in sorted_entries if e["position"] is not None]
    if len(positions) >= 2 and positions[0] > positions[-1]:
        sorted_entries = list(reversed(sorted_entries))
    else:
        instances = [e["instance"] for e in sorted_entries if e["instance"] is not None]
        if len(instances) >= 2 and instances[0] > instances[-1]:
            sorted_entries = list(reversed(sorted_entries))

    return [str(e["path"]) for e in sorted_entries]


def find_best_input(root_dir: str) -> str:
    """Pick the most plausible CT volume inside *root_dir*."""
    for pattern in ("**/*.nii", "**/*.nii.gz"):
        nifti_candidates = sorted(glob.glob(os.path.join(root_dir, pattern), recursive=True))
        if nifti_candidates:
            return nifti_candidates[0]

    candidates = []
    for dirpath, _, _ in os.walk(root_dir):
        if any(seg.startswith(".") for seg in dirpath.split(os.sep)):
            continue
        if looks_like_dicom_dir(dirpath):
            file_count = sum(1 for _ in glob.iglob(os.path.join(dirpath, "**", "*"), recursive=True))
            candidates.append((file_count, dirpath))
    if not candidates:
        raise RuntimeError("Could not find DICOM or NIfTI data in the extracted content")
    candidates.sort(reverse=True)
    return candidates[0][1]


@contextmanager
def prepare_input(input_path: str):
    """Yield (kind, path) for the CT input and clean up temporary files."""
    temp_dir = None
    try:
        if os.path.isdir(input_path) and not is_archive(input_path):
            root = input_path
        else:
            temp_dir = tempfile.mkdtemp(prefix="ct_unpack_")
            unpack_archive_any(input_path, temp_dir)
            root = temp_dir
        best = find_best_input(root)
        if best.lower().endswith((".nii", ".nii.gz")):
            yield "nifti_file", best
        else:
            yield "dicom_dir", best
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def ensure_scalar_3d_float(image: sitk.Image) -> sitk.Image:
    """Cast to 3D float volume, dropping vector/time dimensions if needed."""
    if image.GetNumberOfComponentsPerPixel() > 1:
        image = sitk.VectorIndexSelectionCast(image, 0)
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        size[-1] = 0
        image = sitk.Extract(image, size, [0, 0, 0, 0])
    if image.GetDimension() != 3:
        raise RuntimeError(f"Expected 3D volume, got {image.GetDimension()}D")
    if image.GetPixelID() != sitk.sitkFloat32:
        image = sitk.Cast(image, sitk.sitkFloat32)
    return image


def load_image_from_input(kind: str, path: str) -> Tuple[sitk.Image, Dict[str, str]]:
    if kind == "nifti_file":
        image = sitk.ReadImage(path)
        meta = {
            "study_uid": "",
            "series_uid": Path(path).stem,
            "source": "nifti",
        }
        return image, meta
    if kind == "dicom_dir":
        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        warn_state = sitk.ProcessObject_GetGlobalWarningDisplay()
        if warn_state:
            sitk.ProcessObject_SetGlobalWarningDisplay(False)
        try:
            series_ids = reader.GetGDCMSeriesIDs(path)
        finally:
            if warn_state:
                sitk.ProcessObject_SetGlobalWarningDisplay(True)
        best_meta: Dict[str, str] | None = None
        best_image = None
        best_count = -1
        if series_ids:
            for sid in series_ids:
                file_names = reader.GetGDCMSeriesFileNames(path, sid)
                if not file_names:
                    continue
                if len(file_names) <= best_count:
                    continue
                reader.SetFileNames(file_names)
                try:
                    image_candidate = reader.Execute()
                except RuntimeError:
                    continue
                best_count = len(file_names)
                best_image = image_candidate
                try:
                    study_uid = reader.GetMetaData(0, "0020|000d")
                except Exception:
                    study_uid = ""
                try:
                    series_uid_val = reader.GetMetaData(0, "0020|000e")
                except Exception:
                    series_uid_val = sid
                best_meta = {
                    "study_uid": study_uid,
                    "series_uid": series_uid_val,
                    "source": "dicom",
                }
            if best_image is not None:
                return best_image, best_meta or {"study_uid": "", "series_uid": "", "source": "dicom"}
        dicom_files = collect_dicom_files(path)
        if not dicom_files:
            raise RuntimeError("DICOM series not found")
        ordered_files = _order_dicom_series(dicom_files)
        reader.SetFileNames(ordered_files)
        image = reader.Execute()
        try:
            study_uid = reader.GetMetaData(0, "0020|000d")
        except Exception:
            study_uid = ""
        try:
            series_uid_val = reader.GetMetaData(0, "0020|000e")
        except Exception:
            series_uid_val = ""
        meta = {
            "study_uid": study_uid,
            "series_uid": series_uid_val or Path(path).name,
            "source": "dicom",
        }
        return image, meta
    raise ValueError(f"Unknown source kind: {kind}")


def image_to_hu(image: sitk.Image) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float], Tuple[float, ...]]:
    """Convert SimpleITK image to HU numpy array with spacing/origin/direction."""
    image = ensure_scalar_3d_float(image)
    array_hu = sitk.GetArrayFromImage(image).astype(np.int16)  # [z, y, x]
    spacing_xyz = tuple(float(s) for s in image.GetSpacing())
    origin = tuple(float(o) for o in image.GetOrigin())
    direction = tuple(float(d) for d in image.GetDirection())
    return array_hu, spacing_xyz[::-1], origin, direction  # return spacing as (z, y, x)


def _zyx_to_xyz(spacing_zyx: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])


def _xyz_to_zyx(spacing_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0])


def resample_image_to_spacing(
    image: sitk.Image,
    new_spacing_xyz: Tuple[float, float, float],
    interpolator: int,
    default_value: float = 0.0,
) -> sitk.Image:
    image = ensure_scalar_3d_float(image)
    original_spacing = tuple(float(s) for s in image.GetSpacing())
    original_size = image.GetSize()
    new_size = [
        max(1, int(round(osz * ospc / nspc)))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing_xyz)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing_xyz)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(image)


def resample_to_reference(
    image: sitk.Image, reference: sitk.Image, interpolator: int, default_value: float = 0.0
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(reference.GetSpacing())
    resampler.SetSize(reference.GetSize())
    resampler.SetOutputDirection(reference.GetDirection())
    resampler.SetOutputOrigin(reference.GetOrigin())
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(image)



# ---------------------------------------------------------------------------
# Segmentation utilities
# ---------------------------------------------------------------------------


def _voxel_volume_mm3(spacing_zyx: Tuple[float, float, float]) -> float:
    return float(spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2])


def _ball_structure(radius_mm: float, spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    min_spacing = max(1e-3, min(spacing_zyx))
    radius_vox = max(1, int(round(radius_mm / min_spacing)))
    return morphology.ball(radius_vox)


def _keep_largest(mask: np.ndarray, max_components: int = 1) -> np.ndarray:
    if not mask.any():
        return mask
    labeled = measure.label(mask, connectivity=1)
    component_sizes = np.bincount(labeled.ravel())
    if len(component_sizes) <= 1:
        return mask
    component_ids = np.argsort(component_sizes[1:])[::-1][:max_components] + 1
    return np.isin(labeled, component_ids)


def _remove_small(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 0 or not mask.any():
        return mask
    return morphology.remove_small_objects(mask, min_voxels)


def _per_slice_cleanup(mask: np.ndarray) -> np.ndarray:
    cleaned = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        sl = mask[z]
        if not sl.any():
            continue
        sl = clear_border(sl)
        sl = ndi.binary_fill_holes(sl)
        cleaned[z] = sl
    return cleaned




def _slice_footprint(structure: np.ndarray) -> np.ndarray:
    mid = structure.shape[0] // 2
    footprint = structure[mid]
    return footprint.astype(bool)


def _safe_binary_closing(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    try:
        return ndi.binary_closing(mask, structure=structure)
    except MemoryError:
        footprint = _slice_footprint(structure)
        out = np.zeros_like(mask, dtype=bool)
        for z in range(mask.shape[0]):
            if mask[z].any():
                out[z] = ndi.binary_closing(mask[z], structure=footprint)
        return out


def _safe_binary_opening(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    try:
        return ndi.binary_opening(mask, structure=structure)
    except MemoryError:
        footprint = _slice_footprint(structure)
        out = np.zeros_like(mask, dtype=bool)
        for z in range(mask.shape[0]):
            if mask[z].any():
                out[z] = ndi.binary_opening(mask[z], structure=footprint)
        return out


def build_body_mask(hu: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    body = hu > -650
    body = _safe_binary_closing(body, _ball_structure(3.5, spacing_zyx))
    body = ndi.binary_fill_holes(body)
    body = _remove_small(body, int(max(5_000 / _voxel_volume_mm3(spacing_zyx), 2_000)))
    body = _keep_largest(body, 1)
    return body


def build_thoracic_cavity(bone_mask: np.ndarray, body_mask: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    if not bone_mask.any():
        return body_mask.astype(bool, copy=True)
    shell = _safe_binary_closing(bone_mask, _ball_structure(5.0, spacing_zyx))
    shell = _safe_binary_opening(shell, _ball_structure(2.0, spacing_zyx))
    hull = ndi.binary_fill_holes(shell)
    cavity = hull & body_mask
    return cavity.astype(bool, copy=False)


def _segment_lungs_basic(
    hu: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    body_mask: np.ndarray,
) -> np.ndarray:
    lungs = (hu < -400) & body_mask
    lungs = _per_slice_cleanup(lungs)
    lungs = _safe_binary_opening(lungs, _ball_structure(1.5, spacing_zyx))
    min_voxels = int(max((80_000.0 / _voxel_volume_mm3(spacing_zyx)), 10_000))
    lungs = _remove_small(lungs, min_voxels)
    lungs = _keep_largest(lungs, 2)
    lungs = _safe_binary_closing(lungs, _ball_structure(2.5, spacing_zyx))
    lungs = ndi.binary_fill_holes(lungs)
    return lungs



def _lungs_quality_ok(
    mask: np.ndarray,
    body_mask: np.ndarray,
    body_distance: np.ndarray,
    min_voxels: int,
) -> bool:
    if not mask.any():
        return False
    body_vox = max(int(body_mask.sum()), 1)
    ratio = mask.sum() / float(body_vox)
    if ratio < 0.08:
        return False
    inner = mask & (body_distance >= 3.5)
    if inner.sum() < 0.25 * mask.sum():
        return False
    if mask.sum() < 0.4 * min_voxels:
        return False
    return True


def segment_lungs(
    hu: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    body_mask: np.ndarray,
    thoracic_cavity: Optional[np.ndarray] = None,
    bone_mask: Optional[np.ndarray] = None,
    fast_mode: bool = False,
) -> np.ndarray:
    min_voxels = int(max((80_000.0 / _voxel_volume_mm3(spacing_zyx)), 10_000))
    basic_mask = _segment_lungs_basic(hu, spacing_zyx, body_mask)

    if fast_mode:
        lungs = basic_mask
        if thoracic_cavity is not None:
            lungs &= thoracic_cavity
        lungs = _remove_small(lungs, int(0.25 * min_voxels))
        lungs = _keep_largest(lungs, 2)
        lungs = ndi.binary_fill_holes(lungs)
        if not lungs.any():
            lungs = basic_mask
        return lungs

    body_voxels = max(int(body_mask.sum()), 1)
    basic_ratio = basic_mask.sum() / float(body_voxels)
    if basic_mask.any() and body_mask.shape[0] > 150 and basic_ratio > 0.06 and basic_mask.sum() > 0.35 * min_voxels:
        return basic_mask

    body_distance = ndi.distance_transform_edt(body_mask, sampling=spacing_zyx)
    if _lungs_quality_ok(basic_mask, body_mask, body_distance, min_voxels):
        return basic_mask

    original_candidates = basic_mask.copy()
    lungs = basic_mask.copy()
    bbox_candidates = np.zeros_like(lungs, dtype=bool)
    distance_candidates = np.zeros_like(lungs, dtype=bool)

    labeled = measure.label(lungs, connectivity=1)
    component_ids = range(1, labeled.max() + 1)
    ones = np.ones_like(labeled, dtype=float)
    volumes = ndi.sum(ones, labeled, component_ids)
    centroids = []
    for cid in component_ids:
        centroid = ndi.center_of_mass(ones, labeled, cid)
        if isinstance(centroid, (tuple, list)):
            centroids.append(tuple(float(c) for c in centroid))
        else:
            centroids.append((float(centroid), float(centroid), float(centroid)))

    body_coords = np.where(body_mask)
    if body_coords[0].size > 0:
        y_min, y_max = body_coords[1].min(), body_coords[1].max()
        x_min, x_max = body_coords[2].min(), body_coords[2].max()
    else:
        y_min = x_min = 0
        y_max, x_max = lungs.shape[1] - 1, lungs.shape[2] - 1

    for comp_id, vol, centroid in zip(component_ids, volumes, centroids):
        if vol < 0.10 * min_voxels:
            continue
        cz, cy, cx = centroid
        if not (y_min <= cy <= y_max and x_min <= cx <= x_max):
            continue
        comp_mask = labeled == comp_id
        bbox_candidates |= comp_mask
        dist_vals = body_distance[comp_mask]
        if dist_vals.size == 0 or (dist_vals >= 4.0).mean() < 0.35:
            continue
        distance_candidates |= comp_mask

    if distance_candidates.any():
        lungs = distance_candidates
    elif bbox_candidates.any():
        lungs = bbox_candidates

    lungs = _keep_largest(lungs, 2)
    lungs = _safe_binary_closing(lungs, _ball_structure(2.5, spacing_zyx))
    lungs = ndi.binary_fill_holes(lungs)

    if thoracic_cavity is None and bone_mask is not None:
        thoracic_cavity = build_thoracic_cavity(bone_mask, body_mask, spacing_zyx)

    if thoracic_cavity is not None:
        within_cavity = lungs & thoracic_cavity
        if within_cavity.any() and within_cavity.sum() >= 0.2 * lungs.sum():
            lungs = within_cavity

    if lungs.any():
        inner = lungs & (body_distance >= 4.0)
        if inner.any():
            lungs = inner
    else:
        lungs = original_candidates

    if thoracic_cavity is not None:
        lungs &= thoracic_cavity

    lungs = _remove_small(lungs, int(0.25 * min_voxels))
    lungs = _keep_largest(lungs, 2)
    lungs = ndi.binary_fill_holes(lungs)
    if not lungs.any():
        lungs = original_candidates
    return lungs


def segment_bones(hu: np.ndarray, spacing_zyx: Tuple[float, float, float], body_mask: np.ndarray) -> np.ndarray:
    min_spacing = float(min(spacing_zyx))
    # Исправленные пороговые значения для сегментации костей
    # Кости обычно имеют HU > 150-200, используем более низкие пороги
    density_thr = 200 if min_spacing >= 0.8 else 150
    
    # Первичная сегментация костей с более низким порогом
    bones = (hu > density_thr) & body_mask
    
    # Удаляем мелкие шумы (более мягкое открытие)
    bones = _safe_binary_opening(bones, _ball_structure(0.5, spacing_zyx))
    
    # Удаляем очень мелкие компоненты (более низкий порог)
    min_voxels = int(max(1_000 / _voxel_volume_mm3(spacing_zyx), 500))
    bones = _remove_small(bones, min_voxels)
    
    # Закрываем небольшие разрывы в костях
    bones = _safe_binary_closing(bones, _ball_structure(1.0, spacing_zyx))
    
    # Дополнительная фильтрация по размеру и форме
    if bones.any():
        # Оставляем только крупные компоненты
        bones = _keep_largest(bones, 20)  # Оставляем до 20 крупнейших костных структур
        
        # Удаляем очень мелкие компоненты после морфологических операций
        bones = _remove_small(bones, int(max(2_000 / _voxel_volume_mm3(spacing_zyx), 1_000)))
    
    return bones


def segment_heart(
    hu: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    body_mask: np.ndarray,
    lung_mask: np.ndarray,
    bone_mask: np.ndarray,
) -> np.ndarray:
    voxel_vol = _voxel_volume_mm3(spacing_zyx)
    mediastinum = body_mask & (~lung_mask) & (~bone_mask)
    if not mediastinum.any():
        return np.zeros_like(mediastinum, dtype=bool)

    heart_candidate = (hu > -20) & (hu < 220) & mediastinum

    zz, yy, xx = np.indices(hu.shape)
    lung_coords = np.where(lung_mask)
    if lung_coords[0].size > 0:
        z_pad = int(round(25.0 / max(spacing_zyx[0], 1e-3)))
        z_min = max(0, lung_coords[0].min() - z_pad)
        z_max = min(hu.shape[0], lung_coords[0].max() + z_pad)
        heart_candidate &= (zz >= z_min) & (zz <= z_max)

        x_margin = int(round(18.0 / max(spacing_zyx[2], 1e-3)))
        y_margin = int(round(22.0 / max(spacing_zyx[1], 1e-3)))
        x_min = max(0, lung_coords[2].min() - x_margin)
        x_max = min(hu.shape[2] - 1, lung_coords[2].max() + x_margin)
        y_min = max(0, lung_coords[1].min() - y_margin)
        y_max = min(hu.shape[1] - 1, lung_coords[1].max() + y_margin)
        heart_candidate &= (xx >= x_min) & (xx <= x_max) & (yy >= y_min) & (yy <= y_max)
    else:
        x_center = hu.shape[2] / 2.0
        y_center = hu.shape[1] / 2.0
        heart_candidate &= (np.abs(xx - x_center) <= 0.28 * hu.shape[2]) & (
            np.abs(yy - y_center) <= 0.35 * hu.shape[1]
        )

    body_distance = ndi.distance_transform_edt(body_mask, sampling=spacing_zyx)
    heart_candidate &= body_distance >= 10.0

    fat_mask = (hu <= -60) & mediastinum
    if fat_mask.any():
        heart_candidate &= ~fat_mask

    dense_core = (hu >= 40) & heart_candidate
    if dense_core.any():
        heart_candidate &= ndi.binary_dilation(dense_core, structure=_ball_structure(2.0, spacing_zyx))

    heart_candidate = _safe_binary_closing(heart_candidate, _ball_structure(3.0, spacing_zyx))
    heart_candidate = ndi.binary_fill_holes(heart_candidate)
    heart_candidate = _remove_small(heart_candidate, int(max(6_000.0 / voxel_vol, 2_500)))
    if not heart_candidate.any():
        return heart_candidate

    labeled = measure.label(heart_candidate, connectivity=1)
    if labeled.max() == 0:
        return heart_candidate

    ones = np.ones_like(labeled, dtype=float)
    component_ids = range(1, labeled.max() + 1)
    volumes = ndi.sum(ones, labeled, component_ids)
    mean_hu = ndi.mean(hu.astype(np.float32, copy=False), labeled, component_ids)
    centroids = []
    for cid in component_ids:
        center = ndi.center_of_mass(ones, labeled, cid)
        if isinstance(center, (tuple, list)):
            centroids.append(tuple(float(c) for c in center))
        else:
            centroids.append((float(center), float(center), float(center)))

    lung_center_z = float(lung_coords[0].mean()) if lung_coords[0].size else hu.shape[0] / 2.0
    if lung_coords[2].size > 0:
        x_values = lung_coords[2].astype(float)
        x_mid = float(np.median(x_values))
        left_vals = x_values[x_values <= x_mid]
        right_vals = x_values[x_values >= x_mid]
        left_max = float(left_vals.max()) if left_vals.size else float(x_values.min())
        right_min = float(right_vals.min()) if right_vals.size else float(x_values.max())
    else:
        left_max = 0.0
        right_min = float(hu.shape[2] - 1)

    entries = []
    for cid, vol, mu, centroid in zip(component_ids, volumes, mean_hu, centroids):
        cz, cy, cx = centroid
        between_score = 0
        if left_max < right_min:
            between_score = 0 if (left_max <= cx <= right_min) else 1
        entries.append((cid, vol, mu, cz, between_score))

    entries.sort(key=lambda item: (item[4], -abs(item[3] - lung_center_z), -item[2], -item[1]))

    heart = None
    for cid, vol, _, _, _ in entries:
        if vol * voxel_vol >= 3_000.0:
            heart = labeled == cid
            break
    if heart is None:
        heart = labeled == entries[0][0]

    heart &= hu > -10
    heart = ndi.binary_fill_holes(heart)
    heart = _safe_binary_closing(heart, _ball_structure(2.0, spacing_zyx))
    heart = _remove_small(heart, int(max(6_000.0 / voxel_vol, 2_500)))
    return heart


def refine_segmentation_full(segmentation: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    refined = segmentation.copy()
    voxel_vol_mm3 = _voxel_volume_mm3(spacing_zyx)

    lungs = refined == 1
    if lungs.any():
        lungs = _keep_largest(lungs, 2)
        refined[refined == 1] = 0
        refined[lungs] = 1

    heart = refined == 2
    if heart.any():
        min_vox = int(max(3_500.0 / voxel_vol_mm3, 1_500))
        heart = _remove_small(heart, min_vox)
        heart = ndi.binary_fill_holes(heart)
        refined[refined == 2] = 0
        refined[heart] = 2

    bones = refined == 3
    if bones.any():
        # Более мягкая фильтрация костей - удаляем только очень мелкие компоненты
        bones = _remove_small(bones, int(max(500.0 / voxel_vol_mm3, 200)))
        bones = _safe_binary_closing(bones, _ball_structure(0.8, spacing_zyx))
        refined[refined == 3] = 0
        refined[bones] = 3

    other = refined == 4
    if other.any():
        other = _remove_small(other, int(max(1_000.0 / voxel_vol_mm3, 800)))
        refined[refined == 4] = 0
        refined[other] = 4

    return refined

def _mask_fraction(hu: np.ndarray, mask: np.ndarray, *, mode: str, threshold: float) -> float:
    vox = mask.sum()
    if vox == 0:
        return 0.0
    values = hu[mask]
    if mode == "gt":
        return float((values > threshold).mean())
    if mode == "lt":
        return float((values < threshold).mean())
    raise ValueError(f"Unknown mode: {mode}")



def _analyze_bone_integrity(hu: np.ndarray, bone_mask: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    """Анализ целостности костной структуры."""
    if not bone_mask.any():
        return 0.0
    
    # Анализ непрерывности костной ткани
    labeled_bones = label(bone_mask)[0]
    num_components = labeled_bones.max()
    
    if num_components == 0:
        return 0.0
    
    # Анализ размеров компонентов
    component_sizes = np.bincount(labeled_bones.ravel())[1:]  # Исключаем фон
    total_bone_voxels = bone_mask.sum()
    
    # Оценка целостности на основе распределения размеров
    large_components = np.sum(component_sizes > total_bone_voxels * 0.1)  # >10% от общего объема
    medium_components = np.sum((component_sizes > total_bone_voxels * 0.05) & (component_sizes <= total_bone_voxels * 0.1))
    
    # Нормальная костная структура должна иметь несколько крупных компонентов
    integrity_score = min(1.0, (large_components * 0.5 + medium_components * 0.3) / 3.0)
    
    return float(integrity_score)


def _analyze_bone_density(hu: np.ndarray, bone_mask: np.ndarray) -> Dict[str, Any]:
    """Анализ плотности костной ткани."""
    if not bone_mask.any():
        return {
            'suspicious_lesions': 0,
            'fracture_risk': 0.0,
            'density_variance': 0.0,
            'cortical_integrity': 0.0
        }
    
    bone_hu = hu[bone_mask]
    
    # Анализ литических очагов (низкая плотность)
    lytic_candidates = bone_hu < 100
    lytic_ratio = np.sum(lytic_candidates) / len(bone_hu)
    
    # Анализ бластических очагов (очень высокая плотность)
    blastic_candidates = bone_hu > 1000
    blastic_ratio = np.sum(blastic_candidates) / len(bone_hu)
    
    # Оценка риска переломов
    fracture_risk = lytic_ratio * 0.7 + blastic_ratio * 0.3
    
    # Анализ вариабельности плотности
    density_variance = np.var(bone_hu) / (np.mean(bone_hu) + 1e-6)
    
    # Подсчет подозрительных очагов
    suspicious_lesions = int(lytic_ratio * 10 + blastic_ratio * 5)
    
    return {
        'suspicious_lesions': suspicious_lesions,
        'fracture_risk': float(fracture_risk),
        'density_variance': float(density_variance),
        'cortical_integrity': float(1.0 - fracture_risk)
    }


def _analyze_lung_health(hu: np.ndarray, lung_mask: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> Dict[str, Any]:
    """Базовый анализ здоровья легких."""
    if not lung_mask.any():
        return {
            'volume_abnormal': True,
            'density_abnormal': True,
            'volume_ml': 0.0
        }
    
    voxel_vol_ml = _voxel_volume_mm3(spacing_zyx) / 1000.0
    volume_ml = float(lung_mask.sum()) * voxel_vol_ml
    
    # Анализ плотности легких
    lung_hu = hu[lung_mask]
    dense_fraction = np.sum(lung_hu > -500) / len(lung_hu)
    very_low_fraction = np.sum(lung_hu < -950) / len(lung_hu)
    
    # Критерии аномалий
    volume_abnormal = volume_ml < 2000 or volume_ml > 8000  # Нормальный диапазон
    density_abnormal = dense_fraction > 0.3 or very_low_fraction < 0.1
    
    return {
        'volume_abnormal': volume_abnormal,
        'density_abnormal': density_abnormal,
        'volume_ml': volume_ml,
        'dense_fraction': float(dense_fraction),
        'very_low_fraction': float(very_low_fraction)
    }


def classify_study_bone_focused(
    hu: np.ndarray,
    segmentation: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    summary: Dict[str, Dict[str, float]],
    *,
    fast_mode: bool = False,
    pathology_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    """
    Классификация, основанная на анализе костей и их целостности.
    Фокус на обнаружении патологий костной системы.
    """
    voxel_vol_ml = _voxel_volume_mm3(spacing_zyx) / 1000.0
    
    # Извлечение масок
    lung_mask = segmentation == 1
    bone_mask = segmentation == 3
    
    # Базовые метрики
    lung_volume_ml = float(lung_mask.sum()) * voxel_vol_ml
    bone_volume_ml = float(bone_mask.sum()) * voxel_vol_ml
    
    # Анализ целостности костей
    bone_integrity_score = _analyze_bone_integrity(hu, bone_mask, spacing_zyx)
    
    # Анализ плотности костей
    bone_density_analysis = _analyze_bone_density(hu, bone_mask)
    
    # Анализ легких (базовый)
    lung_analysis = _analyze_lung_health(hu, lung_mask, spacing_zyx)
    
    # Комбинированная оценка
    score = 0.0
    reasons = []
    
    # Критерии по костям (основной фокус) - смягченные
    if bone_integrity_score < 0.3:  # Более строгий порог
        score += 0.4
        reasons.append("bone_integrity_low")
    
    if bone_density_analysis['suspicious_lesions'] > 10:  # Больше очагов для патологии
        score += 0.3
        reasons.append("multiple_bone_lesions")
    
    if bone_density_analysis['fracture_risk'] > 0.8:  # Более высокий риск
        score += 0.2
        reasons.append("high_fracture_risk")
    
    # Критерии по легким (вторичный)
    if lung_analysis['volume_abnormal']:
        score += 0.1
        reasons.append("lung_volume_abnormal")
    
    if lung_analysis['density_abnormal']:
        score += 0.1
        reasons.append("lung_density_abnormal")
    
    # Критерии по патологиям (если доступны)
    if pathology_results is not None:
        pathology_metrics = pathology_results.get('metrics', {})
        
        # Пневмония - серьезная патология
        if pathology_metrics.get('pneumonia', {}).get('present', False):
            score += 0.3
            reasons.append("pneumonia_detected")
        
        # Плевральная жидкость
        if pathology_metrics.get('pleural_effusion', {}).get('present', False):
            score += 0.2
            reasons.append("pleural_effusion")
        
        # Множественные костные очаги
        bone_lesions_count = pathology_metrics.get('bone_lesions', {}).get('num_suspicious', 0)
        if bone_lesions_count > 5:
            score += 0.1
            reasons.append("multiple_bone_lesions")
    
    # Нормализация
    score = min(score, 1.0)
    probability = float(score)
    pathology_flag = int(probability >= 0.5)
    
    if not reasons and pathology_flag == 0:
        reasons.append("normal")
    
    return {
        "probability": probability,
        "pathology_flag": bool(pathology_flag),
        "reasons": reasons,
        "metrics": {
            "lung_volume_ml": lung_volume_ml,
            "bone_volume_ml": bone_volume_ml,
            "bone_integrity_score": bone_integrity_score,
            "bone_density_analysis": bone_density_analysis,
            "lung_analysis": lung_analysis,
        },
    }


def classify_study(
    hu: np.ndarray,
    segmentation: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    summary: Dict[str, Dict[str, float]],
    *,
    fast_mode: bool = False,
) -> Dict[str, object]:
    voxel_vol_ml = _voxel_volume_mm3(spacing_zyx) / 1000.0
    lung_mask = segmentation == 1
    heart_mask = segmentation == 2
    bone_mask = segmentation == 3
    body_mask = segmentation > 0

    thoracic_cavity = None
    if not fast_mode:
        try:
            thoracic_cavity = build_thoracic_cavity(bone_mask, body_mask, spacing_zyx)
        except Exception:
            thoracic_cavity = None

    body_volume_ml = float(body_mask.sum()) * voxel_vol_ml
    if thoracic_cavity is not None and thoracic_cavity.any():
        thoracic_volume_ml = float(thoracic_cavity.sum()) * voxel_vol_ml
    else:
        thoracic_volume_ml = body_volume_ml

    lung_volume_ml = float(summary.get("volumes_ml", {}).get("lungs", 0.0))
    heart_volume_ml = float(summary.get("volumes_ml", {}).get("heart", 0.0))
    bone_volume_ml = float(summary.get("volumes_ml", {}).get("bones", 0.0))

    lung_ratio = float(lung_volume_ml / thoracic_volume_ml) if thoracic_volume_ml > 0 else 0.0
    dense_frac = _mask_fraction(hu, lung_mask, mode="gt", threshold=-500.0)
    very_low_frac = _mask_fraction(hu, lung_mask, mode="lt", threshold=-950.0)

    pleural_air_ml = 0.0
    if not fast_mode and thoracic_cavity is not None:
        pleural_air_mask = thoracic_cavity & (~lung_mask) & (hu < -750)
        pleural_air_ml = float(pleural_air_mask.sum()) * voxel_vol_ml

    if lung_mask.any():
        x_indices = np.arange(lung_mask.shape[2])
        left = lung_mask[:, :, x_indices < lung_mask.shape[2] // 2].sum()
        right = lung_mask[:, :, x_indices >= lung_mask.shape[2] // 2].sum()
        total_lr = left + right
        asymmetry = float(abs(left - right) / total_lr) if total_lr > 0 else 0.0
    else:
        asymmetry = 1.0

    heart_ratio = float(heart_volume_ml / thoracic_volume_ml) if thoracic_volume_ml > 0 else 0.0
    bone_ratio = float(bone_volume_ml / max(body_volume_ml, 1e-3))

    metrics = {
        "thoracic_volume_ml": thoracic_volume_ml,
        "body_volume_ml": body_volume_ml,
        "lung_volume_ml": lung_volume_ml,
        "lung_ratio": lung_ratio,
        "lung_dense_fraction": dense_frac,
        "lung_very_low_fraction": very_low_frac,
        "pleural_air_ml": pleural_air_ml,
        "heart_volume_ml": heart_volume_ml,
        "heart_ratio": heart_ratio,
        "bone_volume_ml": bone_volume_ml,
        "bone_ratio": bone_ratio,
        "lung_lr_asymmetry": asymmetry,
        "fast_mode": bool(fast_mode),
    }

    score = 0.05  # Более консервативный начальный скор
    reasons: List[str] = []

    # Более мягкие критерии для объема легких
    if lung_volume_ml < 800.0 or lung_ratio < 0.20:
        score += 0.40
        reasons.append("lung_volume_low")
    elif lung_ratio > 0.85:
        score += 0.25
        reasons.append("lung_volume_high")
    else:
        score -= 0.10  # Бонус за нормальный объем

    # Более строгие критерии для плотности легких
    if dense_frac > 0.30:
        score += 0.30
        reasons.append("lung_density_high")
    elif dense_frac > 0.15:
        score += 0.10
        reasons.append("mild_lung_density_increase")

    # Критерии для воздушности легких
    if very_low_frac < 0.02 and lung_volume_ml > 1500.0:
        score += 0.15
        reasons.append("air_fraction_low")

    # Плевральный воздух
    if not fast_mode and pleural_air_ml > 100.0:
        score += 0.30
        reasons.append("pleural_air_excess")
    elif not fast_mode and pleural_air_ml > 30.0:
        score += 0.10
        reasons.append("mild_pleural_air")

    # Асимметрия легких
    if asymmetry > 0.50 and lung_volume_ml > 1000.0:
        score += 0.20
        reasons.append("lung_asymmetry")
    elif asymmetry > 0.30 and lung_volume_ml > 1000.0:
        score += 0.05
        reasons.append("mild_lung_asymmetry")

    # Аномалии сердца
    if (not fast_mode) and heart_mask.any():
        if heart_volume_ml < 100.0 or heart_ratio > 0.40 or heart_volume_ml > 2000.0:
            score += 0.15
            reasons.append("mediastinum_soft_tissue_anomaly")
        elif heart_volume_ml < 150.0 or heart_ratio > 0.35:
            score += 0.05
            reasons.append("mild_heart_anomaly")

    # Объем костей - очень мягкие критерии (сегментация костей может быть ненадежной)
    if bone_ratio < 0.003:  # Очень низкий порог
        score += 0.05
        reasons.append("bone_volume_low")
    elif bone_ratio < 0.005:
        score += 0.02
        reasons.append("mild_bone_volume_low")

    score = float(min(max(score, 0.0), 1.0))
    probability = float(score)
    pathology_flag = int(probability >= 0.5)
    if not reasons and pathology_flag == 0:
        reasons.append("no_significant_findings")

    metrics["probability"] = probability
    metrics["pathology_flag"] = pathology_flag

    return {
        "probability": probability,
        "pathology_flag": bool(pathology_flag),
        "reasons": reasons,
        "metrics": metrics,
    }



def segment_chest_organs(
    hu: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    fast_mode: bool = False,
    bones_only: bool = False,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    body_mask = build_body_mask(hu, spacing_zyx)
    bone_mask = segment_bones(hu, spacing_zyx, body_mask)
    
    if bones_only:
        # Режим только костей - сегментируем только кости
        segmentation = np.zeros_like(hu, dtype=np.uint8)
        segmentation[bone_mask] = 3  # Только кости
        
        counts = np.bincount(segmentation.ravel(), minlength=len(LABELS))
        voxel_vol_ml = _voxel_volume_mm3(spacing_zyx) / 1000.0
        volumes_ml = {
            LABELS[i]: float(counts[i] * voxel_vol_ml)
            for i in range(1, len(LABELS))
        }
        diagnostics = {
            "voxel_spacing_mm": {
                "z": float(spacing_zyx[0]),
                "y": float(spacing_zyx[1]),
                "x": float(spacing_zyx[2]),
            },
            "voxel_volume_ml": float(voxel_vol_ml),
            "volumes_ml": volumes_ml,
        }
        return segmentation, diagnostics
    
    # Обычная сегментация всех органов
    thoracic_cavity = None if fast_mode else build_thoracic_cavity(bone_mask, body_mask, spacing_zyx)
    lung_mask = segment_lungs(hu, spacing_zyx, body_mask, thoracic_cavity, bone_mask, fast_mode=fast_mode)
    heart_mask = segment_heart(hu, spacing_zyx, body_mask, lung_mask, bone_mask)
    other_mask = body_mask & (~lung_mask) & (~heart_mask) & (~bone_mask)

    segmentation = np.zeros_like(hu, dtype=np.uint8)
    segmentation[body_mask] = 4
    segmentation[lung_mask] = 1
    segmentation[heart_mask] = 2
    segmentation[bone_mask] = 3
    segmentation[other_mask] = 4

    counts = np.bincount(segmentation.ravel(), minlength=len(LABELS))
    voxel_vol_ml = _voxel_volume_mm3(spacing_zyx) / 1000.0
    volumes_ml = {
        LABELS[i]: float(counts[i] * voxel_vol_ml)
        for i in range(1, len(LABELS))
    }
    diagnostics = {
        "voxel_spacing_mm": {
            "z": float(spacing_zyx[0]),
            "y": float(spacing_zyx[1]),
            "x": float(spacing_zyx[2]),
        },
        "voxel_volume_ml": float(voxel_vol_ml),
        "volumes_ml": volumes_ml,
    }
    return segmentation, diagnostics


def summarize_segmentation(segmentation: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> Dict[str, Dict[str, float]]:
    counts = np.bincount(segmentation.ravel(), minlength=len(LABELS))
    voxel_vol_ml = _voxel_volume_mm3(spacing_zyx) / 1000.0
    volumes_ml = {
        LABELS[i]: float(counts[i] * voxel_vol_ml)
        for i in range(1, len(LABELS))
    }
    return {
        "voxel_spacing_mm": {
            "z": float(spacing_zyx[0]),
            "y": float(spacing_zyx[1]),
            "x": float(spacing_zyx[2]),
        },
        "voxel_volume_ml": float(voxel_vol_ml),
        "volumes_ml": volumes_ml,
    }



# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_segmentation(
    segmentation: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    direction: Tuple[float, ...],
    out_path: str,
) -> None:
    image = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    image.SetSpacing((spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]))
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk.WriteImage(image, out_path)


def save_preview(
    hu: np.ndarray,
    segmentation: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    direction: Tuple[float, ...],
    out_path: str,
    bones_only: bool = False,
) -> None:
    if bones_only:
        # Цветовая карта только для костей
        cmap = mpl_colors.ListedColormap(
            [
                (0.0, 0.0, 0.0, 0.0),  # background
                (0.0, 0.0, 0.0, 0.0),  # lungs (не используется)
                (0.0, 0.0, 0.0, 0.0),  # heart (не используется)
                (1.0, 1.0, 0.2, 0.70),  # bones - более яркий цвет
                (0.0, 0.0, 0.0, 0.0),  # other tissues (не используется)
            ]
        )
    else:
        # Обычная цветовая карта для всех органов
        cmap = mpl_colors.ListedColormap(
            [
                (0.0, 0.0, 0.0, 0.0),
                (0.1, 0.7, 1.0, 0.45),  # lungs
                (1.0, 0.45, 0.2, 0.45),  # heart
                (1.0, 1.0, 0.2, 0.50),  # bones
                (0.7, 0.3, 1.0, 0.35),  # other tissues
            ]
        )

    def render_slice(ax, img_slice, seg_slice, title: str, *, aspect: Optional[str] = None) -> None:
        im_kwargs = {"aspect": aspect} if aspect else {}
        ax.imshow(np.clip(img_slice, -1000, 500), cmap="gray", vmin=-1000, vmax=500, **im_kwargs)
        ax.imshow(seg_slice, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(LABELS) - 1, **im_kwargs)
        ax.set_title(title)
        ax.axis("off")

    image_sitk = sitk.GetImageFromArray(hu.astype(np.float32))
    image_sitk.SetSpacing((spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]))
    image_sitk.SetOrigin(origin)
    image_sitk.SetDirection(direction)

    seg_sitk = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    seg_sitk.CopyInformation(image_sitk)

    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation('LPS')
    image_oriented = orienter.Execute(image_sitk)
    seg_oriented = orienter.Execute(seg_sitk)

    size_xyz = list(image_oriented.GetSize())
    spacing_xyz = list(image_oriented.GetSpacing())

    if size_xyz[2] < 100:
        target_slices = 100
        new_spacing_z = spacing_xyz[2] * size_xyz[2] / target_slices
        new_spacing = (spacing_xyz[0], spacing_xyz[1], float(new_spacing_z))
        new_size = (size_xyz[0], size_xyz[1], target_slices)
        resample_linear = sitk.ResampleImageFilter()
        resample_linear.SetOutputSpacing(new_spacing)
        resample_linear.SetSize([int(v) for v in new_size])
        resample_linear.SetOutputDirection(image_oriented.GetDirection())
        resample_linear.SetOutputOrigin(image_oriented.GetOrigin())
        resample_linear.SetDefaultPixelValue(-1000.0)
        resample_linear.SetInterpolator(sitk.sitkLinear)
        image_oriented = resample_linear.Execute(image_oriented)

        resample_nn = sitk.ResampleImageFilter()
        resample_nn.SetOutputSpacing(new_spacing)
        resample_nn.SetSize([int(v) for v in new_size])
        resample_nn.SetOutputDirection(seg_oriented.GetDirection())
        resample_nn.SetOutputOrigin(seg_oriented.GetOrigin())
        resample_nn.SetDefaultPixelValue(0.0)
        resample_nn.SetInterpolator(sitk.sitkNearestNeighbor)
        seg_oriented = resample_nn.Execute(seg_oriented)

    hu_oriented = sitk.GetArrayFromImage(image_oriented)
    seg_oriented_arr = sitk.GetArrayFromImage(seg_oriented)

    cols = 5
    axial_slices = np.linspace(0, hu_oriented.shape[0] - 1, num=cols, dtype=int)
    coronal_slices = np.linspace(0, hu_oriented.shape[1] - 1, num=cols, dtype=int)
    sagittal_slices = np.linspace(0, hu_oriented.shape[2] - 1, num=cols, dtype=int)
    fig, axes = plt.subplots(3, cols, figsize=(18, 12))

    for idx, z in enumerate(axial_slices):
        render_slice(axes[0, idx], hu_oriented[z], seg_oriented_arr[z], f"Axial z={z}")

    for idx, y in enumerate(coronal_slices):
        img_slice = np.rot90(hu_oriented[:, y, :])
        seg_slice = np.rot90(seg_oriented_arr[:, y, :])
        render_slice(axes[1, idx], img_slice, seg_slice, f"Coronal y={y}", aspect="auto")

    for idx, x in enumerate(sagittal_slices):
        img_slice = np.rot90(hu_oriented[:, :, x])
        seg_slice = np.rot90(seg_oriented_arr[:, :, x])
        render_slice(axes[2, idx], img_slice, seg_slice, f"Sagittal x={x}", aspect="auto")

    for row_axes in axes:
        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_summary(summary: Dict[str, Dict[str, float]], out_path: str) -> None:
    def convert_numpy_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    payload = {
        "labels": LABELS,
        **summary,
    }
    
    # Конвертация numpy типов для JSON сериализации
    payload = convert_numpy_types(payload)
    
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def process_study(
    input_path: str,
    output_root: str,
    fast_mode: bool = False,
    skip_preview: bool = False,
    pathology_analysis: bool = False,
    bones_only: bool = False,
) -> Dict[str, object]:
    start_time = time.perf_counter()
    study_meta: Dict[str, str] = {
        "study_uid": "",
        "series_uid": "",
        "source": "",
    }
    try:
        with prepare_input(input_path) as (kind, path):
            original_image, study_meta = load_image_from_input(kind, path)

        original_image = ensure_scalar_3d_float(original_image)
        large_volume = original_image.GetSize()[2] > 400

        original_spacing_xyz = tuple(float(s) for s in original_image.GetSpacing())
        target_spacing_xyz = compute_target_spacing_xyz(original_spacing_xyz)
        if large_volume:
            target_spacing_xyz = (
                target_spacing_xyz[0],
                target_spacing_xyz[1],
                max(target_spacing_xyz[2], 1.8),
            )
        target_spacing_zyx = _xyz_to_zyx(target_spacing_xyz)
        resample_needed = any(abs(ts - os) > 1e-6 for ts, os in zip(target_spacing_xyz, original_spacing_xyz))
        if resample_needed:
            working_image = resample_image_to_spacing(
                original_image,
                target_spacing_xyz,
                interpolator=sitk.sitkLinear,
                default_value=-1024.0,
            )
        else:
            working_image = original_image

        working_hu, working_spacing_zyx, _, _ = image_to_hu(working_image)
        segmentation_low, low_summary = segment_chest_organs(
            working_hu, working_spacing_zyx, fast_mode=fast_mode, bones_only=bones_only
        )

        seg_working_image = sitk.GetImageFromArray(segmentation_low.astype(np.uint8))
        seg_working_image.CopyInformation(working_image)

        if resample_needed:
            seg_full_image = resample_to_reference(
                seg_working_image,
                original_image,
                interpolator=sitk.sitkNearestNeighbor,
                default_value=0.0,
            )
        else:
            seg_full_image = seg_working_image

        segmentation = sitk.GetArrayFromImage(seg_full_image).astype(np.uint8)

        hu_full, spacing_zyx, origin, direction = image_to_hu(original_image)
        if fast_mode or bones_only:
            summary = summarize_segmentation(segmentation, spacing_zyx)
        else:
            segmentation = refine_segmentation_full(segmentation, spacing_zyx)
            summary = summarize_segmentation(segmentation, spacing_zyx)

        summary["working_spacing_mm"] = {
            "z": float(working_spacing_zyx[0]),
            "y": float(working_spacing_zyx[1]),
            "x": float(working_spacing_zyx[2]),
        }
        summary["target_spacing_mm"] = {
            "z": float(target_spacing_zyx[0]),
            "y": float(target_spacing_zyx[1]),
            "x": float(target_spacing_zyx[2]),
        }
        summary["resampled_for_segmentation"] = bool(resample_needed)
        summary["resample_reason"] = (
            "large_volume"
            if resample_needed and large_volume
            else ("spacing_mismatch" if resample_needed else "none")
        )
        summary["low_res_summary"] = low_summary

        # Определение выходной директории
        base_name = Path(os.path.basename(os.path.normpath(input_path)))
        while base_name.suffix:
            base_name = base_name.with_suffix("")
        study_out_dir = os.path.join(output_root, base_name.name)
        os.makedirs(study_out_dir, exist_ok=True)

        # Инициализация pathology_results
        pathology_results = None
        
        # Анализ патологий (только если не в режиме только костей)
        if pathology_analysis and not bones_only:
            print("Выполняется анализ патологий...")
            try:
                from ct_pathology_analysis import PathologyAnalyzer
                analyzer = PathologyAnalyzer(hu_full, spacing_zyx)
                pathology_results = analyzer.analyze()
                
                # Сохранение результатов анализа патологий
                analyzer.save_metrics(study_out_dir)
                analyzer.save_masks(study_out_dir)
                
                print(f"Результаты анализа патологий сохранены в {study_out_dir}")
            except Exception as e:
                print(f"Ошибка при анализе патологий: {e}")
                pathology_results = {"error": str(e)}
        elif bones_only:
            print("Режим только костей: анализ патологий других органов пропущен")

        if bones_only:
            classification = classify_study_bone_focused(hu_full, segmentation, spacing_zyx, summary, fast_mode=fast_mode, pathology_results=pathology_results)
        else:
            classification = classify_study(hu_full, segmentation, spacing_zyx, summary, fast_mode=fast_mode)
        summary["metrics"] = classification["metrics"]
        summary["classification"] = {
            "probability_of_pathology": classification["probability"],
            "pathology_flag": classification["pathology_flag"],
            "reasons": classification["reasons"],
        }
        summary["metadata"] = {
            "input_path": str(input_path),
            **study_meta,
        }
        summary["processing"] = {
            "fast_mode": bool(fast_mode),
            "skip_preview": bool(skip_preview),
            "large_volume": bool(large_volume),
        }

        preview_path = os.path.join(study_out_dir, "preview.png")
        summary_path = os.path.join(study_out_dir, "summary.json")

        if not skip_preview:
            save_preview(hu_full, segmentation, spacing_zyx, origin, direction, preview_path, bones_only=bones_only)
        save_summary(summary, summary_path)


        print(f"[OK] {input_path} -> {study_out_dir}")
        if bones_only:
            # В режиме только костей показываем только кости
            volume_ml = summary["volumes_ml"].get("bones", 0.0)
            print(f"       {'bones':>18s}: {volume_ml:8.1f} ml")
        else:
            # Обычный режим - показываем все органы кроме сердца
            for label_id, label_name in LABELS.items():
                if label_id == 0 or label_name == "heart":  # Убираем вывод объема сердца
                    continue
                volume_ml = summary["volumes_ml"].get(label_name, 0.0)
                print(f"       {label_name:>18s}: {volume_ml:8.1f} ml")

        probability = float(classification["probability"])
        pathology_flag = int(classification["pathology_flag"])
        if pathology_flag:
            print(
                f"       pathology probability: {probability:.2f} -> PATHOLOGY ({', '.join(classification['reasons'])})"
            )
        else:
            print(f"       pathology probability: {probability:.2f} -> NORMAL")

        duration = time.perf_counter() - start_time
        report_entry: Dict[str, object] = {
            "path_to_study": str(input_path),
            "study_uid": study_meta.get("study_uid", ""),
            "series_uid": study_meta.get("series_uid", ""),
            "probability_of_pathology": probability,
            "pathology": pathology_flag,
            "pathology_reasons": ";".join(classification["reasons"]),
            "processing_status": "Success",
            "time_of_processing": duration,
            "num_slices": int(segmentation.shape[0]),
            "error_message": "",
            "fast_mode": int(fast_mode),
            "skip_preview": int(skip_preview),
        }
        return report_entry
    except Exception as exc:
        duration = time.perf_counter() - start_time
        print(f"[FAIL] {input_path}: {exc}")
        traceback.print_exc()
        report_entry = {
            "path_to_study": str(input_path),
            "study_uid": study_meta.get("study_uid", ""),
            "series_uid": study_meta.get("series_uid", ""),
            "probability_of_pathology": 1.0,
            "pathology": 1,
            "pathology_reasons": f"error:{exc}",
            "processing_status": "Failure",
            "time_of_processing": duration,
            "num_slices": 0,
            "error_message": str(exc),
            "fast_mode": int(fast_mode),
            "skip_preview": int(skip_preview),
        }
        return report_entry


REPORT_COLUMNS = [
    "path_to_study",
    "study_uid",
    "series_uid",
    "probability_of_pathology",
    "pathology",
    "pathology_reasons",
    "processing_status",
    "time_of_processing",
    "num_slices",
    "error_message",
    "fast_mode",
    "skip_preview",
]



def write_report(rows: List[Dict[str, object]], out_path: str) -> None:
    if not rows:
        return
    if Workbook is None:
        raise RuntimeError("openpyxl is required to generate XLSX reports. Install it with 'pip install openpyxl'.")
    wb = Workbook()
    ws = wb.active
    ws.title = "studies"
    ws.append(REPORT_COLUMNS)
    for row in rows:
        ws.append([row.get(col, "") for col in REPORT_COLUMNS])
    wb.save(out_path)


def iter_inputs(input_path: str) -> Iterable[str]:
    if os.path.isdir(input_path):
        entries = []
        for name in sorted(os.listdir(input_path)):
            full = os.path.join(input_path, name)
            if os.path.isdir(full) or is_archive(full):
                entries.append(full)
        if entries:
            return entries
    return [input_path]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chest CT multi-organ segmentation")
    parser.add_argument("--input", required=True, help="Path to a CT study directory or archive")
    parser.add_argument("--out_dir", default="ct_segmentation_out", help="Where to store outputs")
    parser.add_argument("--report_xlsx", default="summary.xlsx", help="Path to the resulting XLSX report")
    parser.add_argument("--fast", action="store_true", help="Enable faster heuristics (less accurate)")
    parser.add_argument("--skip-preview", action="store_true", help="Skip generating preview PNGs")
    parser.add_argument("--pathology-analysis", action="store_true", help="Enable pathology analysis (pleural effusion, calcifications, etc.)")
    parser.add_argument("--bones-only", action="store_true", help="Segment only bones and analyze bone pathology")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inputs = iter_inputs(args.input)
    report_rows: List[Dict[str, object]] = []
    for candidate in inputs:
        row = process_study(candidate, args.out_dir, fast_mode=args.fast, skip_preview=args.skip_preview, pathology_analysis=args.pathology_analysis, bones_only=args.bones_only)
        report_rows.append(row)

    if report_rows:
        write_report(report_rows, args.report_xlsx)
        print(f"[REPORT] saved to {args.report_xlsx}")


if __name__ == "__main__":
    main()











