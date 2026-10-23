import numpy as np


def normalize_ct(ct):
    ct = np.asarray(ct, dtype=np.float32)
    return ((ct - (-1200.0)) / 1800.0).clip(0.0, 1.0).astype(np.float32)


def load_volume(path):
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"ct", "mask"}:
            raise ValueError("Each volume NPZ must contain exactly ct and mask")
        ct = payload["ct"]
        mask = payload["mask"]
    if ct.shape != (64, 64, 64) or mask.shape != ct.shape:
        raise ValueError("CT and mask must both have shape (64, 64, 64)")
    if ct.dtype != np.float32 or mask.dtype != np.uint8:
        raise ValueError("Expected raw-HU float32 ct and uint8 mask")
    if not np.isfinite(ct).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError("CT must be finite and mask must be binary")
    if not mask.any():
        raise ValueError("Nodule mask must be nonempty")
    return ct, mask
