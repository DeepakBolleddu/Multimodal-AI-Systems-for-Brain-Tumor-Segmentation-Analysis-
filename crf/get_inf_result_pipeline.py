import os
import sys
import numpy as np
import nibabel as nib
import torch
import json
import torch.nn.functional as F

# -------------------- Paths & Imports --------------------
MODELS_DIR = "/g/data/ii16/Image/BrainTumorSeg/Models/"
DATA_ROOT  = "/g/data/ii16/Image/BrainTumorSeg/Data/BRATS2021_standardized/"
OUT_ROOT   = "/g/data/ii16/Image/BrainTumorSeg/Inf_Result/og_test_result/PIFNet_test/"
SPLITS_FILE = "/g/data/ii16/Image/BrainTumorSeg/Config/data_splits_updated.json"
PIFNET_PATH = os.path.join(MODELS_DIR, "best_pifnet_model.pth")
VNET_PIPELINE_PATH = os.path.join(MODELS_DIR, "best_vnet_from_pifnet_model.pth")
sys.path.insert(0, MODELS_DIR)
from pif_net import PIFNet
from model import VNet

torch.backends.cudnn.benchmark = True

DEFAULTS = dict(
    data_root=DATA_ROOT,
    out_root=OUT_ROOT,
    pifnet_path=PIFNET_PATH,
    vnet_path=VNET_PIPELINE_PATH,
    in_channels=1, out_channels=1, n_filters=16,
    device_str=None,
    splits_file=SPLITS_FILE,
)

# -------------------- Utils --------------------
def as_DHW(img_nii: nib.Nifti1Image):
    """Convert (H, W, D) to (D, H, W)."""
    arr = img_nii.get_fdata(dtype=np.float32)
    return np.transpose(arr, (2, 0, 1))

def load_four_modalities(t1_p, t1ce_p, t2_p, flair_p):
    t1    = as_DHW(nib.load(t1_p))
    t1ce  = as_DHW(nib.load(t1ce_p))
    t2    = as_DHW(nib.load(t2_p))
    flair = as_DHW(nib.load(flair_p))
    assert t1.shape == t1ce.shape == t2.shape == flair.shape
    # Stack order: [t1, t1ce, t2, flair]
    #vol = np.stack([t1, t1ce, t2, flair], axis=1)  # (D, 4, H, W)
    
    #debug
    vol = np.stack([flair, t1, t1ce, t2], axis=1)
    
    ref = nib.load(flair_p)  # reuse affine/header for saving
    return vol, ref

def load_model(pifnet_path, vnet_path, in_channels, out_channels, n_filters, device):
    pifnet_model = PIFNet()
    pifnet_model.to(device)
    pifnet_model.load_state_dict(torch.load(pifnet_path, map_location=device))
    vnet_model = VNet(in_channels=1, out_channels=1, n_filters=16)
    state = torch.load(vnet_path, map_location=device, weights_only=True)
    vnet_model.load_state_dict(state)
    vnet_model = vnet_model.to(device).eval()
    return pifnet_model, vnet_model

def get_prob_fg_DHW(pifnet_model, vnet_model, vol_DCHW, device):
    """
    vol_DCHW: (D, C, H, W) numpy
    return: (D, H, W) numpy of foreground probability
    """
    vol_NCHWD = torch.from_numpy(np.transpose(vol_DCHW, (1, 2, 3, 0))).unsqueeze(0)  # (1, C, H, W, D)
    vol_NCHWD = vol_NCHWD.to(device=device, dtype=torch.float32)

    # Debug: confirm axis order and eval mode
    print("before model:", vol_NCHWD.shape)   # expect (1, 4, H, W, D)
    print("model.training:", vnet_model.training)  # expect False

    with torch.no_grad():
        pif_output = pifnet_model(vol_NCHWD)                # (1, 1, H, W, D)
        logits = vnet_model(pif_output)
        prob_fg = torch.sigmoid(logits)          # (1, 1, H, W, D)

    # Inspect probability sanity
    print("prob_fg stats: min/mean/max =",
          prob_fg.min().item(), prob_fg.mean().item(), prob_fg.max().item())

    # Compare part
    # prob_fg = (prob_fg > 0.5).float()
    
    prob_fg = prob_fg.squeeze(0).squeeze(0).cpu().numpy()  # (H, W, D)
    
    print(prob_fg.shape)
    
    prob_fg_DHW = np.transpose(prob_fg, (2, 0, 1))         # -> (D, H, W)
    
    print(prob_fg_DHW.shape)
    
    return prob_fg_DHW

# -------------------- Main --------------------
def main(
    data_root=DEFAULTS["data_root"],
    out_root=DEFAULTS["out_root"],
    pifnet_path=DEFAULTS["pifnet_path"],
    vnet_path=DEFAULTS["vnet_path"],
    in_channels=DEFAULTS["in_channels"],
    out_channels=DEFAULTS["out_channels"],
    n_filters=DEFAULTS["n_filters"],
    device_str=DEFAULTS["device_str"],
    splits_file=DEFAULTS["splits_file"],
):
    os.makedirs(out_root, exist_ok=True)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    with open(splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits["test"]

    pifnet_model, vnet_model = load_model(pifnet_path, vnet_path, in_channels, out_channels, n_filters, device)

    for pid in val_ids:
        case_dir = os.path.join(data_root, pid)
        flair_p = os.path.join(case_dir, f"{pid}_flair.nii.gz")
        t1_p    = os.path.join(case_dir, f"{pid}_t1.nii.gz")
        t1ce_p  = os.path.join(case_dir, f"{pid}_t1ce.nii.gz")
        t2_p    = os.path.join(case_dir, f"{pid}_t2.nii.gz")

        vol_DCHW, ref_nii = load_four_modalities(t1_p, t1ce_p, t2_p, flair_p)
        prob_fg_DHW = get_prob_fg_DHW(pifnet_model, vnet_model, vol_DCHW, device)  # (D,H,W)

        prob_fg_HWD = np.transpose(prob_fg_DHW, (1, 2, 0))

        os.makedirs(out_root, exist_ok=True)
        out_path = os.path.join(out_root, f"{pid}_raw_seg.nii.gz")
        nib.save(nib.Nifti1Image(prob_fg_HWD.astype(np.float32), ref_nii.affine, ref_nii.header), out_path)
        print(f"[OK] {pid} -> {out_path}")

if __name__ == "__main__":
    main()
