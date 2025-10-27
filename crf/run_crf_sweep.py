import os
import sys
import itertools
import datetime
from contextlib import redirect_stdout
from importlib.machinery import SourceFileLoader

# ====== CONFIGURATION ======
# Path to your original script (update this to your actual path)
CRF_SCRIPT_PATH = "/g/data/ii16/Image/BrainTumorSeg/CRF_Testing/2DFCCRF.py"

# Root directory where all experiment outputs will be stored
SWEEP_OUT_BASE = "/g/data/ii16/Image/BrainTumorSeg/Inf_Result/SSUNet_crf_sweep/"

# If an output directory already exists:
#   True  -> skip this run (considered finished)
#   False -> overwrite and rerun
RESUME_IF_EXISTS = True

# ---------- Option A: Parameter grid (cartesian product) ----------
# If left empty [], it will not be used.
PARAM_GRID = [
    {
        # You can put multiple data_root/flair_root settings here
        "data_root": [
            "/g/data/ii16/Image/BrainTumorSeg/Inf_Result/SSUNet_subsample/",
        ],
        "flair_root": [
            "/g/data/ii16/Image/BrainTumorSeg/Data/BRATS2021_standardized/",
        ],
        # CRF hyperparameters ,Default: 5,3,3,12,5,5
        "crf_iters": [5], # [5,7,10]
        "sxy_g":     [3], # [2,3,4,5]
        "compat_g":  [3], # [3,5,7]
        "sxy_b":     [12], # [12,15,20]
        "srgb_b":    [5], # [5,8,12]
        "compat_b":  [5,8,12], # [5,8,12]
    }
]

# ---------- Option B: Explicit run list ----------
# If PARAM_GRID is used, leave this empty. Otherwise, list each config manually.
EXPLICIT_RUNS = [
    # Example:
    # {
    #     "data_root":  "/g/data/.../Inf_Result/FusionVNet/",
    #     "flair_root": "/g/data/.../BRATS2021_standardized/",
    #     "crf_iters": 15, "sxy_g": 3, "compat_g": 3, "sxy_b": 12, "srgb_b": 5, "compat_b": 8
    # },
]

# ====== UTILITIES ======

def load_crf_module(py_path):
    """Dynamically load the CRF script from path."""
    name = "crf_runner_target"
    return SourceFileLoader(name, py_path).load_module()

def make_out_dir(base: str, data_root: str, flair_root: str, params: dict) -> str:
    """
    Build a readable and unique output directory name from key parameters.
    Example: FusionVNet__BRATS2021_standardized__it20_g3_b15_rgb5_cb10__20250928-121500
    """
    model_tag = os.path.basename(os.path.normpath(data_root))
    flair_tag = os.path.basename(os.path.normpath(flair_root))

    tag_parts = [
        model_tag,
        flair_tag,
        f"it{params['crf_iters']}",
        f"gg{params['sxy_g']}-cg{params['compat_g']}",
        f"gb{params['sxy_b']}-rb{params['srgb_b']}-cb{params['compat_b']}",
    ]
    dirname = "__".join(tag_parts)
    return os.path.join(base, dirname)

def iter_param_grid(grid_dict: dict):
    """Expand a grid dict into multiple parameter sets (cartesian product)."""
    keys = list(grid_dict.keys())
    vals = [grid_dict[k] if isinstance(grid_dict[k], (list, tuple)) else [grid_dict[k]] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def expand_all_grids(param_grids):
    """Expand multiple grid blocks into parameter sets."""
    for g in param_grids:
        norm = {k: (v if isinstance(v, (list, tuple)) else [v]) for k, v in g.items()}
        for params in iter_param_grid(norm):
            yield params

def run_one(crf_mod, out_root: str, params: dict, log_dir: str):
    """Run a single configuration, redirect stdout to a log file."""
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")

    # Save parameters for reproducibility
    with open(os.path.join(out_root, "sweep_params.txt"), "w", encoding="utf-8") as f:
        f.write(repr(params))

    print(f"[RUN] out_root={out_root}\n      params={params}")
    with open(log_path, "w", encoding="utf-8") as fout, redirect_stdout(fout):
        print(f"[INFO] Starting run with params: {params}")
        # Call the main(...) function from your CRF script
        crf_mod.main(
            data_root = params["data_root"],
            flair_root= params["flair_root"],
            out_root  = out_root,
            crf_iters = params["crf_iters"],
            sxy_g     = params["sxy_g"],
            compat_g  = params["compat_g"],
            sxy_b     = params["sxy_b"],
            srgb_b    = params["srgb_b"],
            compat_b  = params["compat_b"],
        )
        print("[INFO] Finished.")

def main():
    # 1) Load your original CRF script
    if not os.path.exists(CRF_SCRIPT_PATH):
        print(f"[ERR] CRF script not found: {CRF_SCRIPT_PATH}")
        sys.exit(1)
    crf_mod = load_crf_module(CRF_SCRIPT_PATH)

    # 2) Collect all run configurations
    all_runs = []

    if PARAM_GRID:
        for g in PARAM_GRID:
            for params in expand_all_grids([g]):
                all_runs.append(params)

    if EXPLICIT_RUNS:
        all_runs.extend(EXPLICIT_RUNS)

    if not all_runs:
        print("[WARN] No runs configured. Please fill PARAM_GRID or EXPLICIT_RUNS.")
        return

    # 3) Run sequentially
    for i, params in enumerate(all_runs, 1):
        out_dir = make_out_dir(SWEEP_OUT_BASE, params["data_root"], params["flair_root"], params)
        log_dir = os.path.join(out_dir, "_logs")

        if RESUME_IF_EXISTS and os.path.exists(out_dir):
            print(f"[SKIP] ({i}/{len(all_runs)}) exists => {out_dir}")
            continue

        print(f"[{i}/{len(all_runs)}] Running -> {out_dir}")
        run_one(crf_mod, out_dir, params, log_dir)

    print("[ALL DONE]")

if __name__ == "__main__":
    main()
