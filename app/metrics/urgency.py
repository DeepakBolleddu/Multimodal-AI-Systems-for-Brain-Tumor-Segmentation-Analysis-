# app/metrics/urgency.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from scipy.spatial import ConvexHull
from skimage import measure

# Public API -------------------------------------------------------------

def compute_urgency_from_mask(
    mask_data: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    threshold: float = 0.5,
    volume_weight: float = 1.0,
    shape_weight: float = 0.3,
) -> Dict:
    """
    Compute urgency metrics from a tumour mask or probability map.

    Parameters
    ----------
    mask_data : np.ndarray
        3D array. Can be binary (0/1) or probability [0..1].
    voxel_spacing : (sx, sy, sz)
        Spacing in millimetres for each axis.
    threshold : float
        Probability cutoff to binarise if mask_data is not already binary.
    volume_weight, shape_weight : float
        Weights used to aggregate the final urgency score.

    Returns
    -------
    dict
        {
          'final_score': float,
          'urgency_level': 'Low urgency' | 'Medium urgency' | 'High urgency',
          'components': {
             'volume_cm3': float,
             'surface_area_mm2': float,
             'sphericity': float,
             'convex_hull_volume_cm3': float,
             'convexity': float,
             'shape_score': float,
             'volume_score': float,
             'volume_weight': float,
             'shape_weight': float
          }
        }
    """
    # Ensure binary mask
    binary_mask = _binarise_mask(mask_data, threshold=threshold)

    # Volume in cm^3
    num_vox = int(np.count_nonzero(binary_mask))
    voxel_volume_mm3 = float(np.prod(voxel_spacing))
    volume_cm3 = (num_vox * voxel_volume_mm3) / 1000.0

    # Shape metrics
    shape = compute_shape_metrics(binary_mask, voxel_spacing)

    # Aggregate urgency
    volume_score = volume_weight * volume_cm3
    final_score = float(volume_score + shape_weight * shape["shape_score"])
    urgency_level = (
        "High urgency" if final_score >= 50
        else "Medium urgency" if final_score >= 20
        else "Low urgency"
    )

    return {
        "final_score": final_score,
        "urgency_level": urgency_level,
        "components": {
            "volume_cm3": volume_cm3,
            "surface_area_mm2": shape["surface_area_mm2"],
            "sphericity": shape["sphericity"],
            "convex_hull_volume_cm3": shape["convex_hull_volume_cm3"],
            "convexity": shape["convexity"],
            "shape_score": shape["shape_score"],
            "volume_score": volume_score,
            "volume_weight": float(volume_weight),
            "shape_weight": float(shape_weight),
        },
    }


def compute_shape_metrics(binary_mask: np.ndarray,
                          voxel_spacing: Tuple[float, float, float]) -> Dict[str, float]:
    """
    Returns surface/shape metrics with sensible fallbacks for degenerate cases.
    """
    mask_int = (binary_mask.astype(np.uint8) > 0)
    coords = np.argwhere(mask_int)
    n = coords.shape[0]

    if n == 0:
        return {
            "surface_area_mm2": 0.0,
            "sphericity": 1.0,
            "convex_hull_volume_cm3": 0.0,
            "convexity": 1.0,
            "shape_score": 0.0,
        }

    # Volume (mm^3 → cm^3)
    volume_mm3 = float(n * np.prod(voxel_spacing))
    volume_cm3 = volume_mm3 / 1000.0

    # Surface area via marching cubes (triangle mesh area)
    surface_area_mm2 = 0.0
    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask_int.astype(np.float32), level=0.5, spacing=voxel_spacing
        )
        tris = verts[faces]
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        surface_area_mm2 = float(0.5 * np.linalg.norm(cross, axis=1).sum())
    except Exception:
        # very small masks may fail marching_cubes; keep a tiny epsilon
        surface_area_mm2 = 1e-6

    # Convex hull volume (mm^3 → cm^3)
    convex_hull_volume_mm3 = volume_mm3
    if n >= 4:
        try:
            pts_mm = coords * np.asarray(voxel_spacing, dtype=float)
            hull = ConvexHull(pts_mm)
            convex_hull_volume_mm3 = float(hull.volume)
        except Exception:
            pass
    convex_hull_volume_cm3 = convex_hull_volume_mm3 / 1000.0

    # Convexity: hull volume / object volume (>=1 means more irregular)
    convexity = convex_hull_volume_cm3 / max(volume_cm3, 1e-6)

    # Sphericity in [0, 1]
    sphericity = 1.0
    if surface_area_mm2 > 0 and volume_mm3 > 0:
        sphericity = (np.pi ** (1.0 / 3.0)) * ((6.0 * volume_mm3) ** (2.0 / 3.0)) / surface_area_mm2
        sphericity = float(np.clip(sphericity, 0.0, 1.0))

    # Shape risk: penalise deviation from sphere (70%) and convexity irregularity (30%)
    sphericity_penalty = (1.0 - sphericity) * 100.0
    convexity_penalty = max(0.0, convexity - 1.0) * 100.0
    shape_score = float(np.clip(0.7 * sphericity_penalty + 0.3 * convexity_penalty, 0.0, 100.0))

    return {
        "surface_area_mm2": surface_area_mm2,
        "sphericity": sphericity,
        "convex_hull_volume_cm3": convex_hull_volume_cm3,
        "convexity": convexity,
        "shape_score": shape_score,
    }

# Internals -------------------------------------------------------------

def _binarise_mask(mask_data: np.ndarray, *, threshold: float) -> np.ndarray:
    mask_data = np.asarray(mask_data)
    # If already 0/1, leave as-is; else threshold
    if mask_data.dtype.kind in {"b", "i", "u"} and np.array_equal(np.unique(mask_data), [0, 1]):
        return mask_data.astype(np.uint8)
    return (mask_data >= threshold).astype(np.uint8)
