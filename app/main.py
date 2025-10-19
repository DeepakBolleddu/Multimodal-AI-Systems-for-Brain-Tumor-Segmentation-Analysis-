import os
import json
import traceback
from typing import List, Optional
from uuid import uuid4

import numpy as np
import torch
import nibabel as nib
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ✅ package-relative imports (works consistently when app/ is a package)
from metrics.urgency import compute_urgency_from_mask
from model_loader import load_model, run_inference
from utils_nifti import (
    load_nifti_from_bytes, zscore_channelwise, pad_to_divisible,
    safe_normalize_for_surface, marching_mesh, tumor_mesh_from_mask,
    write_obj, mesh_surface_area_mm2, scan_dataset, MODALITY_ORDER_HINT
)


# ---------- Settings ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
os.makedirs(PUBLIC_DIR, exist_ok=True)
load_dotenv(os.path.join(BASE_DIR, ".env"))  # optional .env

OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "static"))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
DATASET_DIR = os.getenv("DATASET_DIR", "")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- App ----------
app = FastAPI(title="Brain Tumor Frontend", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

MODEL = load_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/dataset")
def dataset_info():
    """
    Returns the dataset root and patient_ids,
    plus per-patient modality/mask availability so the frontend can reflect it.
    """
    info = scan_dataset(DATASET_DIR) if DATASET_DIR else {"dataset_dir": DATASET_DIR, "patient_ids": [], "patients": {}}
    return {
        "dataset_dir": info.get("dataset_dir"),
        "modality_hints": MODALITY_ORDER_HINT,
        "patient_ids": info.get("patient_ids", []),
        "patients": info.get("patients", {})
    }

def _assemble_patient_modalities(patient_id: str):
    """
    From DATASET_DIR/patient_id, collect paths for flair/t1ce/t1/t2 (if present).
    Returns (list_paths, list_names) or raises HTTPException if folder missing.
    """
    if not DATASET_DIR:
        raise HTTPException(status_code=400, detail="DATASET_DIR not set in .env")
    pdir = os.path.join(DATASET_DIR, patient_id)
    if not os.path.isdir(pdir):
        raise HTTPException(status_code=404, detail=f"Patient folder not found: {pdir}")
    files = [f for f in os.listdir(pdir) if f.lower().endswith((".nii", ".nii.gz"))]
    if not files:
        raise HTTPException(status_code=404, detail=f"No NIfTI files in {pdir}")

    mod_to_path = {m: None for m in MODALITY_ORDER_HINT}
    for f in files:
        lower = f.lower()
        full = os.path.join(pdir, f)
        for m in MODALITY_ORDER_HINT:
            if m in lower and mod_to_path[m] is None:
                mod_to_path[m] = full

    paths = [p for p in [mod_to_path["flair"], mod_to_path["t1ce"], mod_to_path["t1"], mod_to_path["t2"]] if p]
    names = [os.path.basename(p) for p in paths]
    if not paths:
        raise HTTPException(status_code=400, detail=f"No FLAIR/T1ce/T1/T2 found for patient {patient_id}")
    return paths, names

@app.post("/infer")
async def infer(
    files: List[UploadFile] = File(default=None),
    patient_id: Optional[str] = Query(default=None, description="If provided, read modalities from DATASET_DIR/patient_id")
):
    """
    Inference by:
      - Uploading one or more NIfTI files (FLAIR/T1ce/T1/T2), or
      - Passing ?patient_id=... to load files from DATASET_DIR/patient_id

    Returns:
      - /static/<run_id>/tumor_mask.nii.gz
      - /static/<run_id>/brain_mesh.obj
      - /static/<run_id>/tumor_mesh.obj
      - metrics: voxel size, tumor volume (mm^3), centroid (mm), bbox (mm), surface area (mm^2),
                 and urgency (final_score, urgency_level, components)
    """
    vols = []
    names = []
    # ---------- Load volumes ----------
    if patient_id and (files is None or len(files) == 0):
        # from dataset
        path_list, names = _assemble_patient_modalities(patient_id)
        for p in path_list:
            with open(p, "rb") as fh:
                raw = fh.read()
            v, affine, zooms_xyz = load_nifti_from_bytes(raw, p)
            if v.ndim == 4:
                v = v[..., 0]
            vols.append(v)
        ref_affine = affine
        ref_zooms_xyz = zooms_xyz
    else:
        # from uploads
        if not files:
            raise HTTPException(status_code=400, detail="Upload at least one NIfTI file or provide patient_id.")
        for uf in files:
            name = uf.filename or "uploaded.nii"
            ext = os.path.splitext(name)[1].lower()
            if not (ext in [".nii", ".gz", ".ni"] or name.endswith(".nii.gz")):
                raise HTTPException(status_code=400, detail=f"Unsupported file: {name}. Use .nii/.nii.gz/.ni.")
            raw = await uf.read()
            if len(raw) == 0:
                raise HTTPException(status_code=400, detail=f"Empty file: {name}.")
            v, affine, zooms_xyz = load_nifti_from_bytes(raw, name)
            if v.ndim == 4:
                v = v[..., 0]
            vols.append(v)
            names.append(name)
        ref_affine = affine
        ref_zooms_xyz = zooms_xyz

    # ---------- Stack channels ----------
    if len(vols) == 1:
        # repeat single channel to (4,D,H,W) to satisfy in_channels=4 default
        arr_cdhw = np.stack([vols[0]] * 4, axis=0)
    else:
        # order as [flair, t1ce, t1, t2] if hints in filenames; otherwise keep given order
        def hint_idx(nm):
            n = (nm or "").lower()
            for i, k in enumerate(["flair","t1ce","t1","t2"]):
                if k in n:
                    return i
            return 99
        order = np.argsort([hint_idx(n) for n in names])
        arr_cdhw = np.stack([vols[i] for i in order], axis=0)

    # Normalize and pad
    arr_cdhw = zscore_channelwise(arr_cdhw)
    arr_pad, roi = pad_to_divisible(arr_cdhw, div=16)

    # to torch
    t = torch.from_numpy(arr_pad.astype(np.float32))[None, ...]  # (1,C,Dp,Hp,Wp)
    logits = run_inference(MODEL, t)
    out = logits
    if out.shape[1] > 1:
        out = out[:, 1:2]
    prob = torch.sigmoid(out).squeeze(0).squeeze(0).cpu().numpy()  # (Dp,Hp,Wp)

    # crop back
    zslice, yslice, xslice = roi
    prob = prob[zslice, yslice, xslice]

    # mask
    mask = (prob > 0.5).astype(np.uint8)  # (D,H,W)

    # ---------- Per-run output folder ----------
    run_id = uuid4().hex[:8]
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # ---------- Save NIfTI mask ----------
    mask_name = "tumor_mask.nii.gz"
    mask_path = os.path.join(run_dir, mask_name)
    # our arrays are (Z,Y,X); nibabel expects (X,Y,Z)
    vol_xyz = np.transpose(mask, (2, 1, 0))
    img = nib.Nifti1Image(vol_xyz, ref_affine)
    hdr = img.header
    # zooms are (X,Y,Z); extend with 1.0 for time if needed
    hdr.set_zooms(tuple(float(z) for z in ref_zooms_xyz))
    nib.save(img, mask_path)

    # ---------- Spacing in mm for meshes ----------
    # ref_zooms_xyz: (X,Y,Z) mm. Our arrays are (Z,Y,X) -> spacing_zyx:
    spacing_zyx = (float(ref_zooms_xyz[2]), float(ref_zooms_xyz[1]), float(ref_zooms_xyz[0]))

    # ---------- Brain mesh (scalar from channel 0) ----------
    first_ch = arr_cdhw[0]
    brain_scalar = safe_normalize_for_surface(first_ch)
    brain_verts, brain_faces = marching_mesh(brain_scalar, level=0.25, spacing_zyx=spacing_zyx)

    # ---------- Tumor mesh ----------
    tumor_verts, tumor_faces = tumor_mesh_from_mask(mask, spacing_zyx=spacing_zyx)

    # Save OBJ
    brain_obj_name = "brain_mesh.obj"
    tumor_obj_name = "tumor_mesh.obj"
    brain_obj_path = os.path.join(run_dir, brain_obj_name)
    tumor_obj_path = os.path.join(run_dir, tumor_obj_name)
    write_obj(brain_obj_path, brain_verts, brain_faces)
    write_obj(tumor_obj_path, tumor_verts, tumor_faces)

    # ---------- Base Metrics ----------
    voxel_mm3 = float(np.prod(ref_zooms_xyz)) if ref_zooms_xyz is not None else None
    tumor_voxels = int(mask.sum())
    tumor_volume_mm3 = (tumor_voxels * voxel_mm3) if voxel_mm3 is not None else None

    # centroid + bbox (in mm)
    if tumor_voxels > 0:
        # indices in (Z,Y,X)
        zz, yy, xx = np.where(mask > 0)
        # voxel-space centroid
        cz, cy, cx = float(np.mean(zz)), float(np.mean(yy)), float(np.mean(xx))
        # convert to mm via spacing
        centroid_mm = {
            "x": cx * spacing_zyx[2],
            "y": cy * spacing_zyx[1],
            "z": cz * spacing_zyx[0]
        }
        bbox_mm = {
            "x_min": float(np.min(xx)) * spacing_zyx[2],
            "x_max": float(np.max(xx)) * spacing_zyx[2],
            "y_min": float(np.min(yy)) * spacing_zyx[1],
            "y_max": float(np.max(yy)) * spacing_zyx[1],
            "z_min": float(np.min(zz)) * spacing_zyx[0],
            "z_max": float(np.max(zz)) * spacing_zyx[0],
        }
        surface_area_mm2 = mesh_surface_area_mm2(tumor_verts, tumor_faces)
    else:
        centroid_mm = None
        bbox_mm = None
        surface_area_mm2 = 0.0

    # ---------- Urgency Metrics ----------
    # Load the just-saved NIfTI in (X,Y,Z) to compute urgency with correct voxel spacing.
    mask_img = nib.load(mask_path)
    mask_data_xyz = mask_img.get_fdata()
    zooms_xyz = getattr(mask_img.header, "get_zooms", lambda: (1.0, 1.0, 1.0))()
    voxel_spacing_xyz = tuple(float(z) for z in zooms_xyz[:3]) if len(zooms_xyz) >= 3 else (1.0, 1.0, 1.0)

    urgency = compute_urgency_from_mask(
        mask_data_xyz,
        voxel_spacing=voxel_spacing_xyz,
        threshold=0.5,          # safe if the mask is already 0/1
        volume_weight=1.0,
        shape_weight=0.3,
    )

    # ---------- Combine Metrics ----------
    metrics = {
        "voxel_volume_mm3": voxel_mm3,
        "tumor_voxels": tumor_voxels,
        "tumor_volume_mm3": tumor_volume_mm3,
        "tumor_centroid_mm": centroid_mm,
        "tumor_bbox_mm": bbox_mm,
        "tumor_surface_area_mm2": surface_area_mm2,
        "urgency": urgency
    }

    # Persist metrics alongside artifacts (optional but handy)
    try:
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

    return JSONResponse({
        "run_id": run_id,
        "mask_url": f"/static/{run_id}/{mask_name}",
        "brain_mesh_url": f"/static/{run_id}/{brain_obj_name}",
        "tumor_mesh_url": f"/static/{run_id}/{tumor_obj_name}",
        "metrics": metrics
    })

# ---------- Better error messages during dev ----------
@app.exception_handler(Exception)
async def all_exception_handler(request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print("EXCEPTION:", tb)
    return PlainTextResponse(str(exc), status_code=500)

app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="frontend")
