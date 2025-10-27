import os
import numpy as np
import nibabel as nib
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

# -------------------- Paths --------------------
DATA_ROOT = "/g/data/ii16/Image/BrainTumorSeg/Inf_Result/og_test_result/FusionVNet_test/"
OUT_ROOT  = "/g/data/ii16/Image/BrainTumorSeg/Inf_Result/og_test_result/FusionVNet_crf_test_5_3_2_15_5_5/"


# Naming: BraTS2021_00000_raw_seg.nii.gz
RAW_NAME   = "{pid}_raw_seg.nii.gz"
FLAIR_NAME = "{pid}_flair.nii.gz"

# Update the FLAIR_NAME path to search in the new directory
FLAIR_ROOT = "/g/data/ii16/Image/BrainTumorSeg/Data/BRATS2021_standardized/"

# -------------------- Default CRF Hyperparameters --------------------
DEFAULTS = dict(
    data_root=DATA_ROOT,
    out_root=OUT_ROOT,
    crf_iters=5, sxy_g=3, compat_g=2, sxy_b=15, srgb_b=5, compat_b=5
)

# -------------------- Utils --------------------
def as_DHW(img_nii: nib.Nifti1Image):
    """Convert (H, W, D) to (D, H, W)."""
    arr = img_nii.get_fdata(dtype=np.float32)
    return np.transpose(arr, (2, 0, 1))

def norm_to_u8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def load_flair_DHW(case_dir, pid):
    # Correct the path to search directly in the FLAIR_ROOT directory under the PID folder
    p = os.path.join(FLAIR_ROOT, pid, FLAIR_NAME.format(pid=pid))
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing FLAIR for {pid}: {p}")
    nii = nib.load(p)
    return as_DHW(nii), nii  # Return DHW and reference affine/header

def load_prob_2_DHW(raw_path):
    """
    Supports:
      - (H, W, D) single channel: 0/1 mask or [0,1] probabilities ?? treated as foreground probability p
      - (H, W, D, 2) dual-channel softmax probabilities {bg, fg}
    Returns (2, D, H, W) float32
    """
    nii = nib.load(raw_path)
    arr = nii.get_fdata(dtype=np.float32)

    if arr.ndim == 3:
        # Single channel, foreground probability/mask
        p_fg = np.transpose(arr, (2, 0, 1))  # -> (D,H,W)
        vmin, vmax = float(p_fg.min()), float(p_fg.max())
        if vmax > 1.0 or vmin < 0.0:
            # Not in [0,1], perform linear normalization (rare)
            p_fg = (p_fg - vmin) / (vmax - vmin + 1e-8)
        else:
            p_fg = np.clip(p_fg, 0.0, 1.0)
        p_bg = 1.0 - p_fg
        prob_2 = np.stack([p_bg, p_fg], axis=0).astype(np.float32)  # (2,D,H,W)
        return prob_2

    elif arr.ndim == 4 and arr.shape[-1] == 2:
        # Dual-channel softmax: (H,W,D,2) -> (2,D,H,W)
        prob_2 = np.transpose(arr, (3, 2, 0, 1)).astype(np.float32)
        s = prob_2.sum(axis=0, keepdims=True) + 1e-8
        prob_2 = prob_2 / s
        return prob_2

    else:
        raise ValueError(f"Unsupported raw prob shape {arr.shape}. Expect (H,W,D) or (H,W,D,2).")

def crf_refine_slice(prob2_HW, guide_rgb_HW3, iters, sxy_g, compat_g, sxy_b, srgb_b, compat_b):
    """
    prob2_HW: (2, H, W), softmax-like probabilities summing to 1 along channel axis
    guide_rgb_HW3: (H, W, 3) uint8
    returns (H, W) uint8 labels {0,1}
    """
    C, H, W = prob2_HW.shape
    assert C == 2

    prob2_HW = np.ascontiguousarray(prob2_HW, dtype=np.float32)
    guide_rgb_HW3 = np.ascontiguousarray(guide_rgb_HW3, dtype=np.uint8)

    unary = unary_from_softmax(prob2_HW)
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=sxy_g, compat=compat_g)
    d.addPairwiseBilateral(sxy=sxy_b, srgb=srgb_b, rgbim=guide_rgb_HW3, compat=compat_b)
    Q = d.inference(iters)
    refined = np.array(Q, dtype=np.float32).reshape((C, H, W))
    return np.argmax(refined, axis=0).astype(np.uint8)

def apply_crf(prob_2_DHW, flair_DHW, iters, sxy_g, compat_g, sxy_b, srgb_b, compat_b):
    """
    prob_2_DHW: (2, D, H, W)
    flair_DHW: (D, H, W) used as guidance image
    """
    _, D, H, W = prob_2_DHW.shape
    seg = np.zeros((D, H, W), dtype=np.uint8)
    for z in range(D):
        prob2 = prob_2_DHW[:, z, :, :]
        gray  = norm_to_u8(flair_DHW[z])
        rgb   = np.stack([gray, gray, gray], axis=-1)
        rgb   = np.ascontiguousarray(rgb, dtype=np.uint8)
        seg[z] = crf_refine_slice(prob2, rgb, iters, sxy_g, compat_g, sxy_b, srgb_b, compat_b)
    return seg

# -------------------- Main --------------------
def main(
    data_root=DEFAULTS["data_root"],
    flair_root=FLAIR_ROOT,
    out_root=DEFAULTS["out_root"],
    crf_iters=DEFAULTS["crf_iters"],
    sxy_g=DEFAULTS["sxy_g"],
    compat_g=DEFAULTS["compat_g"],
    sxy_b=DEFAULTS["sxy_b"],
    srgb_b=DEFAULTS["srgb_b"],
    compat_b=DEFAULTS["compat_b"],
):
    os.makedirs(out_root, exist_ok=True)

   
    cases = sorted(f for f in os.listdir(data_root) if f.endswith("_raw_seg.nii.gz"))
    print(f"Found cases: {cases[:5]} ... total={len(cases)}")

    for fname in cases:
        pid = fname.replace("_raw_seg.nii.gz", "")  
        raw_path   = os.path.join(data_root, f"{pid}_raw_seg.nii.gz")
        flair_path = os.path.join(flair_root, pid, f"{pid}_flair.nii.gz")

        print(f"Checking RAW path: {raw_path}")
        print(f"Checking FLAIR path: {flair_path}")

        if not os.path.exists(raw_path):
            print(f"[skip] missing RAW for {pid}")
            continue
        if not os.path.exists(flair_path):
            print(f"[skip] missing FLAIR for {pid}")
            continue

        print(f"Processing {pid} ...")
       
        flair_DHW, ref_nii = load_flair_DHW(FLAIR_ROOT, pid)

        try:
            prob_2_DHW = load_prob_2_DHW(raw_path)
        except Exception as e:
            print(f"[skip] failed to load prob for {pid}: {e}")
            continue

        # CRF refine
        seg_DHW = apply_crf(prob_2_DHW, flair_DHW, crf_iters, sxy_g, compat_g, sxy_b, srgb_b, compat_b)

        seg_HWD = np.transpose(seg_DHW, (1, 2, 0)).astype(np.uint8)
        hdr = nib.Nifti1Header()
        hdr.set_data_dtype(np.uint8)
        hdr['scl_slope'] = 1
        hdr['scl_inter'] = 0

        out_path = os.path.join(out_root, f"{pid}_crf_seg.nii.gz")
        nib.save(nib.Nifti1Image(seg_HWD, affine=ref_nii.affine, header=hdr), out_path)
        print(f"[OK] {pid} -> {out_path}")
    
    # Export parameter settings
    text = f"crf_iters={crf_iters}, sxy_g={sxy_g}, compat_g={compat_g}, sxy_b={sxy_b}, srgb_b={srgb_b}, compat_b={compat_b}"
    setting_path = os.path.join(out_root, "parameter_settings.txt")
    with open(setting_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"The setting has been exported to {out_root} successfully.")

if __name__ == "__main__":

    flair_path = os.path.join(FLAIR_ROOT, "BraTS2021_00000", FLAIR_NAME.format(pid="BraTS2021_00000"))

    print(f"FLAIR exists: {os.path.exists(flair_path)}")
    print(f"Generated FLAIR path: {flair_path}")
    print(f"FLAIR_ROOT: {FLAIR_ROOT}")
    main()
