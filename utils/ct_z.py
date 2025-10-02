#!/usr/bin/env python3
"""
ct_zprofile.py — Slice-wise z-profile and large-scale abnormality score for CT.

Outputs (per slice):
- lung_area_px / mm2 (if PixelSpacing present)
- mean_lung_HU
- frac_dense_m500 (HU > -500 inside lung)  ~ consolidation/atelectasis proxy
- frac_dense_m300 (HU > -300 inside lung)  ~ very dense lung (fluid/effusion contacting lung)
- frac_emph_m950  (HU < -950 inside lung)  ~ emphysema proxy
- LR_asym (|L-R|/max(L,R))                 ~ big unilateral changes
- score ∈ [0,1] (heuristic combination)

Plots: features and score vs z (or slice index if z unknown), with flagged segments.

NOTE: Heuristic lung segmentation via threshold + morphology. Tune thresholds as needed.
"""

import argparse, os, csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Suppress pydicom warnings about invalid UIDs (common in Russian DICOM files)
warnings.filterwarnings('ignore', category=UserWarning, module='pydicom')

import pydicom
from pydicom.filereader import dcmread

from skimage.segmentation import clear_border
from skimage.morphology import binary_closing, remove_small_objects, remove_small_holes, disk
from skimage.measure import label, regionprops


_CT_SOP_UIDS = {
    "1.2.840.10008.5.1.4.1.1.2",      # CT Image
    "1.2.840.10008.5.1.4.1.1.2.1",    # Enhanced CT
    "1.2.840.10008.5.1.4.1.1.2.2",    # Legacy Converted Enhanced CT
}

def _is_ct_like(meta):
    sop = str(getattr(meta, "SOPClassUID", "")).strip()
    modality = str(getattr(meta, "Modality", "")).upper().strip()
    return sop in _CT_SOP_UIDS or modality == "CT"

def dcmread_any(ref, *, stop_before_pixels=False, force=True):
    """Accept a disk path OR a {'kind': 'zip'|'disk', ...} ref."""
    if isinstance(ref, (str, os.PathLike)):
        return dcmread(ref, stop_before_pixels=stop_before_pixels, force=force)
    if isinstance(ref, dict) and ref.get("kind") in ("disk", "zip"):
        if ref["kind"] == "disk":
            return dcmread(ref["path"], stop_before_pixels=stop_before_pixels, force=force)
        # Zip: read whole member to a seekable BytesIO buffer (required by pydicom)
        with zipfile.ZipFile(ref["zip_path"]) as zf, zf.open(ref["member"]) as fh:
            bio = io.BytesIO(fh.read())
            return dcmread(bio, stop_before_pixels=stop_before_pixels, force=force)
    raise TypeError(f"dcmread_any: unsupported ref type {type(ref)}")

def _make_disk_ref(path): return {"kind": "disk", "path": path}
def _make_zip_ref(zpath, member): return {"kind": "zip", "zip_path": zpath, "member": member}

def load_series(path_or_zip, debug=False):
    """
    Returns: dict(series_uid -> list of (meta_dcm, file_ref))
    Works with directories OR .zip (any nesting).
    """
    series = defaultdict(list)
    seen = dicom = ct = 0

    def _consider(meta, ref):
        nonlocal dicom, ct
        dicom += 1
        # Skip DICOMDIR SOP Class
        if str(getattr(meta, "SOPClassUID", "")) == "1.2.840.10008.1.3.10":
            return
        if _is_ct_like(meta):
            uid = getattr(meta, "SeriesInstanceUID", None)
            if uid:
                series[uid].append((meta, ref))
                ct += 1

    if os.path.isdir(path_or_zip):
        for root, _, files in os.walk(path_or_zip):
            for f in files:
                fp = os.path.join(root, f); seen += 1
                try:
                    meta = dcmread(fp, stop_before_pixels=True, force=True)
                except Exception:
                    continue
                _consider(meta, _make_disk_ref(fp))
        if debug:
            print(f"[load_series_any] dir seen={seen} dicom={dicom} ct_like={ct} series={len(series)}")
        return series

    if os.path.isfile(path_or_zip) and zipfile.is_zipfile(path_or_zip):
        zpath = path_or_zip
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if info.is_dir(): 
                    continue
                # Skip obvious non-DICOM small files quickly
                if info.file_size < 132: 
                    continue
                seen += 1
                try:
                    with zf.open(info) as fh:
                        bio = io.BytesIO(fh.read())
                        meta = dcmread(bio, stop_before_pixels=True, force=True)
                except Exception:
                    continue
                _consider(meta, _make_zip_ref(zpath, info.filename))
        if debug:
            print(f"[load_series_any] zip='{os.path.basename(path_or_zip)}' seen={seen} dicom={dicom} ct_like={ct} series={len(series)}")
        return series

    raise FileNotFoundError(f"Not a directory or zip: {path_or_zip}")



# ----------------- DICOM helpers -----------------
def try_get(d, name, default=None):
    return getattr(d, name, default) if hasattr(d, name) else default

def to_hu(dcm):
    arr = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept

def sort_key_meta(d):
    ipp = try_get(d, "ImagePositionPatient", None)
    if ipp is not None:
        try:
            return float(ipp[2])
        except Exception:
            pass
    inn = try_get(d, "InstanceNumber", None)
    if inn is not None:
        try:
            return int(inn)
        except Exception:
            pass
    return 0.0

def get_frame_z_enhanced(d, frame_idx):
    try:
        pffg = d.PerFrameFunctionalGroupsSequence[frame_idx]
        for key in ("PlanePositionSequence", "PlanePositionPatientSequence"):
            if hasattr(pffg, key):
                seq = getattr(pffg, key)
                if seq and hasattr(seq[0], "ImagePositionPatient"):
                    ipp = seq[0].ImagePositionPatient
                    return float(ipp[2])
    except Exception:
        pass
    return None



import os, io, zipfile
from collections import defaultdict
import pydicom
from pydicom.filereader import dcmread

# -------- unified file-ref helpers --------
def _make_disk_ref(path):
    return {"kind": "disk", "path": path}

def _make_zip_ref(zip_path, member):
    return {"kind": "zip", "zip_path": zip_path, "member": member}

def read_meta_from_ref(ref):
    """Read DICOM header only (stop_before_pixels=True) from disk or zip."""
    if ref["kind"] == "disk":
        return dcmread(ref["path"], stop_before_pixels=True, force=True)
    else:
        with zipfile.ZipFile(ref["zip_path"]) as zf, zf.open(ref["member"]) as fh:
            bio = io.BytesIO(fh.read())
            return dcmread(bio, stop_before_pixels=True, force=True)

def read_full_from_ref(ref):
    """Read full DICOM (with pixels) from disk or zip."""
    if ref["kind"] == "disk":
        d = dcmread(ref["path"], force=True)
        _ = d.pixel_array  # trigger decode early; surfaces errors sooner
        return d
    else:
        with zipfile.ZipFile(ref["zip_path"]) as zf, zf.open(ref["member"]) as fh:
            bio = io.BytesIO(fh.read())
            d = dcmread(bio, force=True)
            _ = d.pixel_array
            return d

def _try_get(d, name, default=None):
    return getattr(d, name, default) if hasattr(d, name) else default



def z_sort_and_stack(dcms):
    """Return (stack [N,H,W] float32, z_list [N], px_mm (row,col))"""
    dcms_sorted = sorted(dcms, key=lambda t: sort_key_meta(t[0]))
    imgs, zs = [], []
    row_spacing = col_spacing = None

    for d_meta, fp in dcms_sorted:
        d = dcmread_any(fp, force=True)
        ps = try_get(d, "PixelSpacing", None)
        if ps is not None and len(ps) >= 2:
            row_spacing = float(ps[0]); col_spacing = float(ps[1])

        hu = to_hu(d)
        if hu.ndim == 2:
            z = None
            ipp = try_get(d, "ImagePositionPatient", None)
            if ipp is not None:
                try: z = float(ipp[2])
                except Exception: pass
            imgs.append(hu.astype(np.float32))
            zs.append(z)
        elif hu.ndim == 3:  # Enhanced
            n = hu.shape[0]
            z_list = [get_frame_z_enhanced(d, i) for i in range(n)]
            order = list(range(n))
            if all(z is not None for z in z_list):
                order = [i for i, _ in sorted(enumerate(z_list), key=lambda t: t[1])]
            for i in order:
                imgs.append(hu[i].astype(np.float32))
                zs.append(z_list[i])
        else:
            continue

    if len(imgs) == 0:
        raise SystemExit("No slices decoded.")

    if sum(z is not None for z in zs) > len(zs)//2:
        order = np.argsort([z if z is not None else 0.0 for z in zs])
        imgs = [imgs[i] for i in order]
        zs   = [zs[i]   for i in order]

    stack = np.stack(imgs, axis=0)
    px_mm = (row_spacing, col_spacing) if (row_spacing and col_spacing) else (None, None)
    return stack, zs, px_mm

# ----------------- Lung segmentation -----------------
def segment_lungs(hu_slice):
    """
    Heuristic per-slice lung mask:
    - threshold air-like voxels (< -500 HU)
    - remove components touching the border (outside air)
    - close, remove small bits, fill holes
    - keep up to two largest components as lungs; split L/R by centroid x
    Returns (lung_mask, left_mask, right_mask) booleans.
    """
    m = hu_slice < -500  # air-ish
    m = clear_border(m)
    m = binary_closing(m, disk(3))
    m = remove_small_objects(m, min_size=1000)
    m = remove_small_holes(m, area_threshold=1000)

    if m.any():
        lab = label(m)
        props = sorted(regionprops(lab), key=lambda r: r.area, reverse=True)
        keep_labels = [p.label for p in props[:2]]  # at most two lungs
        m = np.isin(lab, keep_labels)
        # split L/R by centroid x
        left_mask = np.zeros_like(m, bool)
        right_mask = np.zeros_like(m, bool)
        if props:
            # Assign each kept component to left/right by centroid
            H, W = m.shape
            for p in props[:2]:
                comp = (lab == p.label)
                if p.centroid[1] < W/2:
                    left_mask |= comp
                else:
                    right_mask |= comp
        else:
            left_mask = m
            right_mask = np.zeros_like(m, bool)
    else:
        left_mask = np.zeros_like(m, bool)
        right_mask = np.zeros_like(m, bool)

    lung_mask = m
    return lung_mask, left_mask, right_mask

# ----------------- Feature extraction -----------------
def slice_features(hu, px_mm):
    lung, L, R = segment_lungs(hu)
    inside = lung.sum()
    if inside == 0:
        return {
            "lung_area_px": 0,
            "lung_area_mm2": None,
            "mean_lung_HU": np.nan,
            "frac_dense_m500": np.nan,
            "frac_dense_m300": np.nan,
            "frac_emph_m950": np.nan,
            "LR_asym": np.nan,
        }

    vals = hu[lung]
    meanHU = float(np.mean(vals))
    frac_dense_m500 = float(np.mean(vals > -500.0))
    frac_dense_m300 = float(np.mean(vals > -300.0))
    frac_emph_m950  = float(np.mean(vals < -950.0))

    L_area = L.sum()
    R_area = R.sum()
    if max(L_area, R_area) > 0:
        LR_asym = float(abs(L_area - R_area) / max(L_area, R_area))
    else:
        LR_asym = np.nan

    row_mm, col_mm = px_mm
    if row_mm is not None and col_mm is not None:
        area_mm2 = inside * row_mm * col_mm
    else:
        area_mm2 = None

    return {
        "lung_area_px": int(inside),
        "lung_area_mm2": area_mm2,
        "mean_lung_HU": meanHU,
        "frac_dense_m500": frac_dense_m500,
        "frac_dense_m300": frac_dense_m300,
        "frac_emph_m950":  frac_emph_m950,
        "LR_asym": LR_asym,
    }

def moving_average(y, w):
    if w is None or w <= 1:
        return y
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(y, kernel, mode="same")

def minmax01(x):
    x = np.asarray(x, float)
    m = np.nanmin(x)
    M = np.nanmax(x)
    if not np.isfinite(m) or not np.isfinite(M) or abs(M - m) < 1e-9:
        return np.zeros_like(x, float)
    return (x - m) / (M - m)

def compute_score(feat):
    """
    Combine features into a simple abnormality score ∈ [0,1].
    Higher = more suspicious (dense lung, reduced air, asymmetry).
    """
    # Normalize each (ignore NaNs via nanmin/nanmax in minmax01)
    dense500_n = minmax01(feat["frac_dense_m500"])
    meanHU_n   = minmax01(feat["mean_lung_HU"])
    asym_n     = minmax01(feat["LR_asym"])
    # 'less air' is abnormal → invert lung area norm
    area_n     = minmax01(feat["lung_area_px"])
    less_air_n = 1.0 - area_n

    score = 0.45 * dense500_n + 0.25 * meanHU_n + 0.10 * asym_n + 0.10 * less_air_n
    score = np.clip(score, 0.0, 1.0)
    return score

# ----------------- Main analysis -----------------
def analyze(stack, zs, px_mm, smooth=None):
    N = stack.shape[0]
    # X-axis: true z if known for all; else slice index
    have_all_z = all(z is not None for z in zs)
    if have_all_z:
        z_axis = np.array([float(z) for z in zs], dtype=float)
    else:
        z_axis = np.arange(N, dtype=float)

    # Per-slice features
    feats = {
        "lung_area_px": np.zeros(N, int),
        "lung_area_mm2": np.array([np.nan]*N, float),
        "mean_lung_HU": np.array([np.nan]*N, float),
        "frac_dense_m500": np.array([np.nan]*N, float),
        "frac_dense_m300": np.array([np.nan]*N, float),
        "frac_emph_m950": np.array([np.nan]*N, float),
        "LR_asym": np.array([np.nan]*N, float),
    }
    for i in range(N):
        f = slice_features(stack[i], px_mm)
        for k in feats.keys():
            feats[k][i] = f[k]

    # Smooth selected curves for display (does not affect raw CSV)
    sm = smooth if (smooth and smooth > 1) else None
    disp = {k: moving_average(v, sm) if k in (
        "mean_lung_HU","frac_dense_m500","frac_dense_m300","frac_emph_m950","LR_asym","lung_area_px"
    ) else v for k, v in feats.items()}

    score = compute_score(feats)
    score_disp = moving_average(score, sm)

    return z_axis, feats, disp, score, score_disp, have_all_z

def plot_profiles(z, disp, score_disp, have_all_z, score_thresh):
    xlabel = "z (mm)" if have_all_z else "slice index"
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Lung area
    axes[0].plot(z, disp["lung_area_px"], lw=1)
    axes[0].set_ylabel("Lung area (px)")
    axes[0].set_title("Lung area vs z")

    # Density fractions
    axes[1].plot(z, disp["frac_dense_m500"], lw=1, label="HU > -500 (lung)")
    axes[1].plot(z, disp["frac_dense_m300"], lw=1, label="HU > -300 (lung)")
    axes[1].plot(z, disp["frac_emph_m950"], lw=1, label="HU < -950 (lung)")
    axes[1].legend(loc="best")
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("Lung density fractions vs z")

    # Asymmetry & mean HU
    ax2a = axes[2]
    ax2b = ax2a.twinx()
    ax2a.plot(z, disp["LR_asym"], lw=1, label="LR asym (left axis)")
    ax2a.set_ylabel("LR asym")
    ax2b.plot(z, disp["mean_lung_HU"], lw=1, color="tab:orange", label="Mean HU (right axis)")
    ax2b.set_ylabel("Mean HU")
    axes[2].set_title("Left-Right asymmetry & mean HU vs z")

    # Score with threshold shading
    axes[3].plot(z, score_disp, lw=1.5)
    axes[3].axhline(score_thresh, ls="--", lw=1, color="tab:red")
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("Abnormality score")
    axes[3].set_title("Large-scale abnormality score vs z")
    axes[3].set_xlabel(xlabel)

    # Shade runs above threshold
    above = score_disp >= score_thresh
    if np.any(above):
        # find contiguous segments
        idx = np.where(above)[0]
        starts = [idx[0]]
        ends = []
        for i in range(1, len(idx)):
            if idx[i] != idx[i-1] + 1:
                ends.append(idx[i-1])
                starts.append(idx[i])
        ends.append(idx[-1])
        for s, e in zip(starts, ends):
            axes[3].axvspan(z[s], z[e], color="tab:red", alpha=0.15)

    fig.tight_layout()
    plt.show()

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(description="CT z-profile & large-scale abnormality scoring.")
    ap.add_argument("dicom_dir", help="Path to directory with DICOM CT series.")
    ap.add_argument("--smooth", type=int, default=1, help="Moving-average window (slices) for display.")
    ap.add_argument("--score-thresh", type=float, default=0.7, help="Threshold to flag suspicious z-runs.")
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV path to save per-slice features and score.")
    return ap.parse_args()

def main():
    args = parse_args()
    series = load_series(args.dicom_dir)
    if not series:
        raise SystemExit("No CT series found.")
    uid = next(iter(series.keys()))
    stack, zs, px_mm = z_sort_and_stack(series[uid])
    print(f"Loaded {stack.shape[0]} slices. Pixel spacing (row,col) mm: {px_mm}")

    z, feats, disp, score, score_disp, have_all_z = analyze(stack, zs, px_mm, smooth=args.smooth)

    if args.csv:
        keys = ["lung_area_px","lung_area_mm2","mean_lung_HU","frac_dense_m500","frac_dense_m300","frac_emph_m950","LR_asym","score"]
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["z"] + keys)
            for i in range(len(z)):
                w.writerow([z[i]] + [feats[k][i] if k != "score" else score[i] for k in keys])
        print(f"Wrote CSV: {args.csv}")
    plot_profiles(z, disp, score_disp, have_all_z, args.score_thresh)

if __name__ == "__main__":
    main()

