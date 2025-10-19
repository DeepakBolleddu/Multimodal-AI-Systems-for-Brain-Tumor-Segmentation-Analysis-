import os
import io
import re
from typing import List, Tuple, Optional, Dict
import numpy as np
import nibabel as nib
from skimage import measure

# Modality name hints (case-insensitive substring match)
MODALITY_ORDER_HINT = ["flair", "t1ce", "t1", "t2"]
MASK_HINTS = ["seg", "mask"]

def load_nifti_from_bytes(raw: bytes, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Windows-safe temp-file loading. Ensures we delete AFTER get_fdata().
    Returns (vol_zyx, affine, zooms_xyz)
    """
    import tempfile
    if filename.endswith(".nii.gz") or filename.endswith(".gz"):
        suffix = ".nii.gz"
    else:
        suffix = ".nii"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        img = nib.load(tmp_path)
        data = img.get_fdata().astype(np.float32)  # triggers file reads
        affine = img.affine
        zooms = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # Standardize to (Z,Y,X[,C])
    if data.ndim == 3:
        vol = np.transpose(data, (2, 1, 0))
    elif data.ndim == 4:
        vol = np.transpose(data, (2, 1, 0, 3))
    else:
        raise ValueError(f"Unsupported NIfTI shape: {data.shape}")

    return vol, affine, zooms

def zscore_channelwise(arr: np.ndarray) -> np.ndarray:
    """
    arr: (C,D,H,W)
    """
    out = arr.copy()
    for c in range(out.shape[0]):
        ch = out[c]
        mu = float(np.mean(ch))
        sd = float(np.std(ch)) + 1e-6
        out[c] = (ch - mu) / sd
    return out

def pad_to_divisible(arr: np.ndarray, div: int = 16) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    """
    arr: (C,D,H,W) -> pad to multiples of 'div'
    returns padded array and ROI slices to crop back
    """
    C, D, H, W = arr.shape
    def grow(n):
        r = n % div
        return n if r == 0 else n + (div - r)
    Dp, Hp, Wp = grow(D), grow(H), grow(W)
    out = np.zeros((C, Dp, Hp, Wp), dtype=arr.dtype)
    out[:, :D, :H, :W] = arr
    return out, (slice(0, D), slice(0, H), slice(0, W))

def safe_normalize_for_surface(vol_zyx: np.ndarray) -> np.ndarray:
    """
    Robust 0..1 normalization for a scalar field. Returns zeros if collapsed range.
    """
    vol = vol_zyx.astype(np.float32)
    p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
    denom = (p99 - p1)
    if not np.isfinite(denom) or denom <= 1e-6:
        return np.zeros_like(vol, dtype=np.float32)
    out = (vol - p1) / (denom + 1e-6)
    return np.clip(out, 0.0, 1.0)

def marching_mesh(volume_zyx: np.ndarray, level: float = 0.25, spacing_zyx=(1.0,1.0,1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mesh from scalar field (Z,Y,X). Returns empty arrays if invalid.
    """
    v = volume_zyx.astype(np.float32)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= 1e-6:
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.int32)
    if not (vmin <= level <= vmax):
        level = vmin + 0.25 * (vmax - vmin)
        if level <= vmin or level >= vmax:
            return np.zeros((0,3), np.float32), np.zeros((0,3), np.int32)
    verts, faces, _, _ = measure.marching_cubes(v, level=level, spacing=spacing_zyx)
    return verts, faces.astype(np.int32)

def tumor_mesh_from_mask(mask_zyx: np.ndarray, spacing_zyx=(1.0,1.0,1.0)) -> Tuple[np.ndarray, np.ndarray]:
    mask = (mask_zyx.astype(np.uint8) > 0)
    if mask.sum() == 0:
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.int32)
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0.5, spacing=spacing_zyx)
    return verts, faces.astype(np.int32)

def write_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

def mesh_surface_area_mm2(verts: np.ndarray, faces: np.ndarray) -> float:
    """
    Compute surface area in units of verts (mm if verts are in mm).
    """
    if verts.size == 0 or faces.size == 0:
        return 0.0
    try:
        return float(measure.mesh_surface_area(verts, faces))
    except Exception:
        # manual area if needed
        v = verts
        f = faces
        p0 = v[f[:,0]]
        p1 = v[f[:,1]]
        p2 = v[f[:,2]]
        a = np.linalg.norm(p1 - p0, axis=1)
        b = np.linalg.norm(p2 - p1, axis=1)
        c = np.linalg.norm(p0 - p2, axis=1)
        s = 0.5 * (a + b + c)
        area = np.sqrt(np.clip(s*(s-a)*(s-b)*(s-c), 0, None))
        return float(np.sum(area))

# --------------------------
# Dataset scanning utilities
# --------------------------
def _contains_any(s: str, keys: List[str]) -> bool:
    s = s.lower()
    return any(k in s for k in keys)

def detect_modality_from_name(name: str) -> Optional[str]:
    n = name.lower()
    for key in MODALITY_ORDER_HINT:
        if key in n:
            return key
    return None

def is_mask_name(name: str) -> bool:
    return _contains_any(name, MASK_HINTS)

def scan_dataset(dataset_dir: str) -> Dict:
    """
    Scans dataset root:
      dataset_dir/
        PatientA/
          ... nifti files ...
        PatientB/
          ...
    Returns a dict with patient_ids and per-patient file availability.
    """
    out = {
        "dataset_dir": dataset_dir,
        "patient_ids": [],
        "patients": {}  # id -> dict(modalities: {flair, t1ce, t1, t2}, has_mask: bool)
    }
    if not dataset_dir or not os.path.isdir(dataset_dir):
        return out
    for entry in sorted(os.listdir(dataset_dir)):
        pdir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(pdir):
            continue
        files = [f for f in os.listdir(pdir) if f.lower().endswith((".nii", ".nii.gz"))]
        if not files:
            continue
        info = {
            "modalities": {m: None for m in MODALITY_ORDER_HINT},
            "has_mask": False
        }
        for f in files:
            fpath = os.path.join(pdir, f)
            mod = detect_modality_from_name(f)
            if mod and info["modalities"].get(mod) is None:
                info["modalities"][mod] = fpath
            if is_mask_name(f):
                info["has_mask"] = True
        out["patient_ids"].append(entry)
        out["patients"][entry] = info
    return out
