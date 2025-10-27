#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, csv, json
from collections import OrderedDict
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage

# =========================
# Fixed paths
# =========================
INF_ROOT  = "/g/data/ii16/Image/BrainTumorSeg/Inf_Result/"
GT_ROOT   = "/g/data/ii16/Image/BrainTumorSeg/Data/BRATS2021_standardized/"
SAVE_ROOT = "/g/data/ii16/Image/BrainTumorSeg/CRF_Testing/evaluate_Result/"

# File name suffix patterns
RAW_SUFFIXES = ("_raw_seg.nii.gz", "_seg.nii.gz")   # RAW may use either naming
CRF_SUFFIXES = ("_crf_seg.nii.gz", "_seg.nii.gz")   # CRF may use either naming

# -------------------------
# IO helpers
# -------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_seg(path: str) -> np.ndarray:
    """Load a NIfTI file and binarize: values > 0.5 are treated as foreground."""
    nii = nib.load(path)
    arr = nii.get_fdata(dtype=np.float32)
    return (arr > 0.5).astype(np.uint8)

def resize_to_match(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize to target_shape with nearest-neighbor interpolation."""
    factors = [t / s for t, s in zip(target_shape, arr.shape)]
    return ndimage.zoom(arr, zoom=factors, order=0)

def get_gt_path(pid: str):
    """
    Return GT path for a given PID.
    Prefer <pid>_seg.nii.gz; fall back to <pid>_crf_seg.nii.gz (with a warning).
    """
    p1 = os.path.join(GT_ROOT, pid, f"{pid}_seg.nii.gz")
    if os.path.exists(p1):
        return p1, None
    p2 = os.path.join(GT_ROOT, pid, f"{pid}_crf_seg.nii.gz")
    if os.path.exists(p2):
        return p2, f"[WARN] Using fallback GT '{pid}_crf_seg.nii.gz' (unusual as GT)."
    return None, None

def parse_pid(filename: str) -> str | None:
    """Extract a BraTS PID like 'BraTS2021_00001' from a filename."""
    m = re.search(r"(BraTS2021_\d{5})", filename)
    return m.group(1) if m else None

def collect_prediction_files(group_dir: str, suffixes: tuple[str, ...]) -> dict:
    """
    Recursively collect prediction files under group_dir that end with any of the given suffixes.
    Return a dict {pid: file_path}. If multiple files for the same PID exist, the last one wins.
    """
    paths = {}
    for f in glob.glob(os.path.join(group_dir, "**", "*.nii.gz"), recursive=True):
        if not f.endswith(suffixes):
            continue
        pid = parse_pid(os.path.basename(f))
        if pid:
            # If there are duplicates for the same PID, keep the latest (overwrite by dict behavior).
            paths[pid] = f
    if not paths:
        print(f"[HINT] No predictions with suffix {suffixes} under {group_dir}")
    return OrderedDict(sorted(paths.items(), key=lambda x: x[0]))

# -------------------------
# Per-case CSV helpers
# -------------------------
CASE_HEADER = [
    "pid", "tp", "fp", "fn",
    "precision", "recall", "f1", "iou", "dice"
]

def read_cases_csv(path: str) -> dict:
    """Read an existing cases.csv and return {pid: row_dict}."""
    if not os.path.exists(path):
        return {}
    data = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row.get("pid")
            if not pid:
                continue
            # Safe conversions
            try:
                tp = int(row.get("tp", 0)); fp = int(row.get("fp", 0)); fn = int(row.get("fn", 0))
                pr = float(row.get("precision", 0)); rc = float(row.get("recall", 0))
                f1 = float(row.get("f1", 0)); iou = float(row.get("iou", 0)); dice = float(row.get("dice", 0))
            except Exception:
                continue
            data[pid] = dict(pid=pid, tp=tp, fp=fp, fn=fn,
                             precision=pr, recall=rc, f1=f1, iou=iou, dice=dice)
    return data

def write_cases_csv(path: str, rows: list[dict]) -> None:
    """Write rows to cases.csv with a fixed header."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CASE_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CASE_HEADER})

# -------------------------
# Metric helpers
# -------------------------
def pr_from_counts(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def rc_from_counts(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    if denom > 0:
        return (2 * tp) / denom
    # All negatives on both sides -> treat as perfect F1=1.0
    return 1.0

def iou_from_counts(tp, fp, fn):
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 1.0

# -------------------------
# Evaluation core (with PID skipping)
# -------------------------
def evaluate_group(group_dir: str, save_dir: str, suffixes: tuple[str, ...]):
    """
    Evaluate one group directory:
      - Read save_dir/cases.csv and skip PIDs already evaluated.
      - Evaluate only new PIDs, then merge with previous results.
      - Compute micro/macro metrics on all cases (old + new).
      - Return aggregated results and write back cases.csv.
    """
    ensure_dir(save_dir)
    cases_csv = os.path.join(save_dir, "cases.csv")

    existing = read_cases_csv(cases_csv)     # {pid: row}
    done_pids = set(existing.keys())

    pid2file = collect_prediction_files(group_dir, suffixes)
    if not pid2file:
        return None

    new_rows = []
    TP_sum = FP_sum = FN_sum = 0  # streaming counts for "newly evaluated" cases only

    total = len(pid2file)
    for i, (pid, pred_path) in enumerate(pid2file.items(), 1):
        if pid in done_pids:
            if i % 50 == 0 or i == total:
                print(f"[INFO] {os.path.basename(group_dir)} - skipped {i}/{total} (already in cases.csv)")
            continue

        gt_path, warn = get_gt_path(pid)
        if gt_path is None:
            print(f"[WARN] Missing GT for {pid}. Skipped.")
            continue
        if warn:
            print(warn)

        try:
            pred = load_seg(pred_path)
            gt   = load_seg(gt_path)
        except Exception as e:
            print(f"[WARN] Load error for {pid}: {e}. Skipped.")
            continue

        if pred.shape != gt.shape:
            print(f"[WARN] Shape mismatch {pid}: pred={pred.shape}, gt={gt.shape}. Resizing pred.")
            pred = resize_to_match(pred, gt.shape)

        p = pred.ravel().astype(np.uint8, copy=False)
        g = gt.ravel().astype(np.uint8, copy=False)

        tp = int(np.sum((g == 1) & (p == 1)))
        fp = int(np.sum((g == 0) & (p == 1)))
        fn = int(np.sum((g == 1) & (p == 0)))

        TP_sum += tp; FP_sum += fp; FN_sum += fn

        pr  = pr_from_counts(tp, fp)
        rc  = rc_from_counts(tp, fn)
        f1  = f1_from_counts(tp, fp, fn)
        iou = iou_from_counts(tp, fp, fn)
        dice = f1  # identical for binary segmentation

        new_rows.append(dict(pid=pid, tp=tp, fp=fp, fn=fn,
                             precision=pr, recall=rc, f1=f1, iou=iou, dice=dice))

        if i % 5 == 0 or i == total:
            print(f"[INFO] {os.path.basename(group_dir)} - processed {i}/{total}")

    # Merge old + new (new rows overwrite same PIDs if any; usually none because we skip)
    merged = list(existing.values())
    by_pid = {r["pid"]: r for r in merged}
    for r in new_rows:
        by_pid[r["pid"]] = r
    merged = list(sorted(by_pid.values(), key=lambda x: x["pid"]))

    # Write cases.csv
    write_cases_csv(cases_csv, merged)
    print(f"[SAVED] cases -> {cases_csv} (total cases: {len(merged)}, newly added: {len(new_rows)})")

    if not merged:
        return None

    # Compute micro/macro on the merged set
    TP = sum(r["tp"] for r in merged)
    FP = sum(r["fp"] for r in merged)
    FN = sum(r["fn"] for r in merged)

    micro_p   = pr_from_counts(TP, FP)
    micro_r   = rc_from_counts(TP, FN)
    micro_f1  = f1_from_counts(TP, FP, FN)
    micro_iou = iou_from_counts(TP, FP, FN)

    macro_p    = float(np.mean([r["precision"] for r in merged])) if merged else 0.0
    macro_r    = float(np.mean([r["recall"]    for r in merged])) if merged else 0.0
    macro_f1   = float(np.mean([r["f1"]        for r in merged])) if merged else 0.0
    macro_iou  = float(np.mean([r["iou"]       for r in merged])) if merged else 0.0
    macro_dice = float(np.mean([r["dice"]      for r in merged])) if merged else 0.0

    return {
        "micro": (micro_p, micro_r, micro_f1, micro_iou),
        "macro": (macro_p, macro_r, macro_f1, macro_iou, macro_dice),
        "n_cases": len(merged),
        "cases_csv": cases_csv
    }

# -------------------------
# Saving helpers
# -------------------------
def model_dir(model: str) -> str:
    """Return the directory for a model under SAVE_ROOT; create if missing."""
    d = os.path.join(SAVE_ROOT, model)
    ensure_dir(d);  return d

def group_dir(model: str, group_name: str) -> str:
    """Return the directory for a group under a model; create if missing."""
    d = os.path.join(model_dir(model), group_name)
    ensure_dir(d);  return d

def save_group_eval(model: str, group_name: str, res: dict) -> str:
    """Write per-group aggregate metrics to <group>/eval.csv."""
    out = os.path.join(group_dir(model, group_name), "eval.csv")
    (mp, mr, mf1, miou) = res["micro"]
    (MaP, MaR, MaF1, MaIoU, MaDice) = res["macro"]
    n = res["n_cases"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group", "Cases", "Micro_Precision", "Micro_Recall", "Micro_F1", "Micro_IoU",
                    "Macro_Precision", "Macro_Recall", "Macro_F1", "Macro_IoU", "Macro_Dice"])
        w.writerow([group_name, n, f"{mp:.6f}", f"{mr:.6f}", f"{mf1:.6f}", f"{miou:.6f}",
                    f"{MaP:.6f}", f"{MaR:.6f}", f"{MaF1:.6f}", f"{MaIoU:.6f}", f"{MaDice:.6f}"])
    print(f"[SAVED] eval  -> {out}")
    return out

def save_model_comparison(model: str, results_by_group: dict, baseline_group: str) -> str:
    """
    Save a comparison CSV against the specified baseline group:
    <Model>/_comparison_vs_<baseline_group>.csv
    """
    out = os.path.join(model_dir(model), f"_comparison_vs_{baseline_group}.csv")
    raw = results_by_group.get(baseline_group)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group",
                    "Micro_P","Micro_R","Micro_F1","Micro_IoU",
                    "Macro_P","Macro_R","Macro_F1","Macro_IoU","Macro_Dice",
                    f"ΔMicro_F1(vs {baseline_group})", f"ΔMacro_Dice(vs {baseline_group})"])
        for gname, res in sorted(results_by_group.items()):
            if gname == baseline_group:
                continue
            (mp, mr, mf1, miou) = res["micro"]
            (MaP, MaR, MaF1, MaIoU, MaDice) = res["macro"]
            d_micro_f1 = d_macro_dice = ""
            if raw:
                d_micro_f1   = f"{(mf1   - raw['micro'][2]):+.6f}"
                d_macro_dice = f"{(MaDice - raw['macro'][4]):+.6f}"
            w.writerow([gname, f"{mp:.6f}", f"{mr:.6f}", f"{mf1:.6f}", f"{miou:.6f}",
                        f"{MaP:.6f}", f"{MaR:.6f}", f"{MaF1:.6f}", f"{MaIoU:.6f}", f"{MaDice:.6f}",
                        d_micro_f1, d_macro_dice])
    print(f"[SAVED] comparison -> {out}")
    return out

def save_global_summary(results_all: dict) -> str:
    """Save a global summary across all models/groups."""
    out = os.path.join(SAVE_ROOT, "_ALL_groups_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model","Group","Micro_P","Micro_R","Micro_F1","Micro_IoU",
                    "Macro_P","Macro_R","Macro_F1","Macro_IoU","Macro_Dice","Cases"])
        for (model, gname), res in sorted(results_all.items()):
            (mp, mr, mf1, miou) = res["micro"]
            (MaP, MaR, MaF1, MaIoU, MaDice) = res["macro"]
            n = res["n_cases"]
            w.writerow([model, gname, f"{mp:.6f}", f"{mr:.6f}", f"{mf1:.6f}", f"{miou:.6f}",
                        f"{MaP:.6f}", f"{MaR:.6f}", f"{MaF1:.6f}", f"{MaIoU:.6f}", f"{MaDice:.6f}", n])
    print(f"[SAVED] global summary -> {out}")
    return out

# -------------------------
# Sweep discovery
# -------------------------
def discover_models_and_groups(inf_root: str):
    """
    Discover sweep containers and groups.

    Returns a dict:
    {
      model_name: {
         "sweep_dir": <path>,
         "raw_dir":   <path or None>,
         "crf_groups": { group_name: <path>, ... }
      }, ...
    }

    Logic: only scan <model>_crf_sweep containers and treat <model>_subsample
    as the RAW folder inside that container. All other subfolders are CRF groups.
    """
    result = {}
    for name in sorted(os.listdir(inf_root)):
        if not name.endswith("_crf_sweep"):
            continue
        sweep_dir = os.path.join(inf_root, name)
        if not os.path.isdir(sweep_dir):
            continue

        model = name[:-len("_crf_sweep")]  # model name without suffix
        raw_dir_name = f"{model}_subsample"
        raw_dir = os.path.join(sweep_dir, raw_dir_name)
        if not os.path.isdir(raw_dir):
            print(f"[WARN] RAW folder missing for {model}: {raw_dir}")

        crf_groups = {}
        for sub in sorted(os.listdir(sweep_dir)):
            p = os.path.join(sweep_dir, sub)
            if not os.path.isdir(p):
                continue
            if sub == raw_dir_name:
                continue
            crf_groups[sub] = p

        result[model] = dict(sweep_dir=sweep_dir, raw_dir=raw_dir if os.path.isdir(raw_dir) else None,
                             crf_groups=crf_groups)
    return result

# -------------------------
# Main orchestrator
# -------------------------
def main():
    ensure_dir(SAVE_ROOT)
    results_all = {}

    layout = discover_models_and_groups(INF_ROOT)
    if not layout:
        print(f"[ERROR] No '*_crf_sweep' found under {INF_ROOT}")
        return

    for model, info in layout.items():
        print(f"\n===== MODEL: {model} =====")
        model_results = {}

        # 1) RAW
        if info["raw_dir"]:
            gname = os.path.basename(info["raw_dir"])
            save_dir = group_dir(model, gname)
            res = evaluate_group(info["raw_dir"], save_dir, RAW_SUFFIXES)
            if res:
                save_group_eval(model, gname, res)
                model_results[gname] = res
                results_all[(model, gname)] = res
            else:
                print(f"[WARN] RAW has no valid pairs: {info['raw_dir']}")
        else:
            print(f"[WARN] No RAW dir for {model}")

        # 2) CRF groups
        for gname, gpath in info["crf_groups"].items():
            print(f"[GROUP] {model}/{gname} -> {gpath}")
            save_dir = group_dir(model, gname)
            res = evaluate_group(gpath, save_dir, CRF_SUFFIXES)
            if res:
                save_group_eval(model, gname, res)
                model_results[gname] = res
                results_all[(model, gname)] = res
            else:
                print(f"[Skip] {gname} has no valid prediction/GT pairs.")

        # 3) Comparison against RAW (only if a RAW group exists)
        if model_results:
            raw_candidates = [g for g in model_results.keys() if g.endswith("_subsample")]
            if raw_candidates:
                baseline_group = raw_candidates[0]
                save_model_comparison(model, model_results, baseline_group=baseline_group)

    # 4) Global summary
    save_global_summary(results_all)
    print("\n[DONE] All evaluations written under:", SAVE_ROOT)


if __name__ == "__main__":
    main()
