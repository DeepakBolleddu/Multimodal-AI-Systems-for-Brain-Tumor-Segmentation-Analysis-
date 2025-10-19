import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

# Fixed defaults – adjust if you need different tradeoffs
CRF_PARAMS_DEFAULT = {
    "crf_iters": 5,   # typical 5~10
    "sxy_g": 3,       # gaussian spatial kernel
    "compat_g": 3,    # gaussian weight
    "sxy_b": 12,      # bilateral spatial kernel
    "srgb_b": 5,      # bilateral intensity kernel (we use FLAIR as 1 channel)
    "compat_b": 12     # bilateral weight
}

def _densecrf_refine_slice_multiclass(probs_hwC: np.ndarray, guide_hwC: np.ndarray, params: dict) -> np.ndarray:
    """
    Refine a 2D slice with DenseCRF (multiclass).
    probs_hwC: (H,W,C) softmax probabilities including background
    guide_hwC: (H,W,G) guidance image; here G=1 (FLAIR only)
    return: (H,W) int64 labels
    """
    H, W, C = probs_hwC.shape
    # Ensure arrays are C-contiguous for pydensecrf
    probs_hwC = np.ascontiguousarray(probs_hwC, dtype=np.float32)
    unary = unary_from_softmax(probs_hwC.transpose(2, 0, 1)).reshape(C, -1)
    unary = np.ascontiguousarray(unary, dtype=np.float32)

    crf = dcrf.DenseCRF2D(W, H, C)
    crf.setUnaryEnergy(unary)

    crf.addPairwiseEnergy(
        create_pairwise_gaussian(sdims=(params["sxy_g"], params["sxy_g"]), shape=(H, W)),
        compat=params["compat_g"]
    )

    g = np.ascontiguousarray(guide_hwC, dtype=np.float32)
    # normalize each guidance channel to [0,255] for stability
    for c in range(g.shape[2]):
        x = g[..., c]
        mn, mx = x.min(), x.max()
        if mx > mn:
            x = (x - mn) / (mx - mn)
        g[..., c] = x * 255.0
    g = np.ascontiguousarray(g.astype("uint8"))

    crf.addPairwiseEnergy(
        create_pairwise_bilateral(
            sdims=(params["sxy_b"], params["sxy_b"]),
            schan=([params["srgb_b"]] * g.shape[2]),
            img=g, chdim=2
        ),
        compat=params["compat_b"]
    )

    Q = crf.inference(params["crf_iters"])
    Q = np.array(Q).reshape(C, H, W).transpose(1, 2, 0)  # (H,W,C)
    return np.argmax(Q, axis=-1).astype(np.int64)

def refine_volume_axial_binary(prob_zyx: np.ndarray, flair_zyx: np.ndarray, params: dict | None = None) -> np.ndarray:
    """
    Apply 2D DenseCRF per axial slice for binary tumor segmentation.
    prob_zyx:  (Z,Y,X) sigmoid tumor probability in [0,1]
    flair_zyx: (Z,Y,X) FLAIR guidance
    return: (Z,Y,X) uint8 mask {0,1}
    """
    if params is None:
        params = CRF_PARAMS_DEFAULT

    # Ensure input arrays are C-contiguous
    prob_zyx = np.ascontiguousarray(prob_zyx, dtype=np.float32)
    flair_zyx = np.ascontiguousarray(flair_zyx, dtype=np.float32)
    
    Z, Y, X = prob_zyx.shape
    out = np.zeros((Z, Y, X), dtype=np.uint8)

    for z in range(Z):
        p = prob_zyx[z]                      # (Y,X)
        # build 2-class softmax probs: [background, tumor]
        probs_hw2 = np.stack([1.0 - p, p], axis=-1)  # (Y,X,2)
        probs_hw2 = np.ascontiguousarray(probs_hw2, dtype=np.float32)
        guide_hw1 = np.ascontiguousarray(flair_zyx[z][..., None], dtype=np.float32)  # (Y,X,1)
        # fallback if degenerate slice
        if not np.isfinite(probs_hw2).all() or np.allclose(probs_hw2.std(), 0):
            out[z] = (p > 0.5).astype(np.uint8)
            continue
        lbl = _densecrf_refine_slice_multiclass(probs_hw2, guide_hw1, params)  # (Y,X) in {0,1}
        out[z] = (lbl == 1).astype(np.uint8)

    return out
