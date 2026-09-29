"""
Dataset Loader Module
======================
PyTorch Dataset classes for the three AI/ML tasks (detection, classification,
prediction), plus a synthetic-data generator so the training/inference
pipeline can be tested end-to-end before real satellite data is wired in.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# IMD cyclone intensity categories as they actually appear in the real INCYDE
# labels (D/DD/CS/SCS/VSCS/ESCS/SuCS - no "Low Pressure Area" since these are
# all already-named, tracked storms). Index order must match CATEGORY_MAP in
# scripts/build_incyde_manifest.py.
CYCLONE_CATEGORIES = [
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm",
]

REAL_IMG_SIZE = 128  # resize target for real INCYDE jpgs


def load_image_any(path):
    """
    Loads either a synthetic .npy patch (C, H, W) or a real INCYDE .jpg
    (grayscale IR imagery) resized to a fixed size and returned as (1, H, W).
    """
    if str(path).endswith(".npy"):
        return np.load(path).astype(np.float32)

    from PIL import Image
    img = Image.open(path).convert("L").resize((REAL_IMG_SIZE, REAL_IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    return arr[np.newaxis, :, :]  # (1, H, W) - single-channel real IR imagery


class CycloneImageDataset(Dataset):
    """
    Used for both detection (binary: cyclone patch vs. no-cyclone patch) and
    classification (multi-class category) tasks.

    Expects a manifest CSV/DataFrame with columns:
        image_path, label   (label: 0/1 for detection, 0..7 for classification)
    Each image_path points to a saved .npy patch of shape (C, H, W), where the
    channels are stacked IR1/IR2/WV/VIS (and optionally SST/wind) bands.
    """

    def __init__(self, manifest_df, task="detection", transform=None):
        self.manifest = manifest_df.reset_index(drop=True)
        self.task = task
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        patch = load_image_any(row["image_path"])
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)

        if self.transform:
            patch = self.transform(patch)

        label = int(row["label"])
        return torch.from_numpy(patch), torch.tensor(label, dtype=torch.long)


class CycloneSequenceDataset(Dataset):
    """
    Used for the prediction task (future track + intensity). Each sample is a
    sequence of `seq_len` past frames/observations and the corresponding
    future targets: (delta_lat, delta_lon, wind_speed, pressure) for the next
    `horizon` time steps.

    Expects a manifest with columns:
        sequence_path, target_path
    sequence_path -> .npy of shape (T, C, H, W)
    target_path   -> .npy of shape (horizon, 4)  [dlat, dlon, wind, pressure]
    """

    def __init__(self, manifest_df, seq_len=8, horizon=4):
        self.manifest = manifest_df.reset_index(drop=True)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        seq = np.load(row["sequence_path"]).astype(np.float32)
        target = np.load(row["target_path"]).astype(np.float32)

        seq = (seq - seq.mean()) / (seq.std() + 1e-6)
        return torch.from_numpy(seq), torch.from_numpy(target)


# ---------------------------------------------------------------------------
# Synthetic data generator (for pipeline testing before real data is ready)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(out_dir, n_samples=200, patch_size=64,
                                channels=4, task="detection",
                                seq_len=8, horizon=4, seed=42):
    """
    Creates a small synthetic dataset on disk and returns a manifest
    DataFrame, so `train.py` and `inference_pipeline.py` can be run and
    verified before real INSAT/reanalysis data is plugged in.
    """
    import os
    import pandas as pd

    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    if task in ("detection", "classification"):
        n_classes = 2 if task == "detection" else len(CYCLONE_CATEGORIES)
        for i in range(n_samples):
            patch = rng.normal(size=(channels, patch_size, patch_size)).astype(np.float32)
            label = int(rng.integers(0, n_classes))
            path = os.path.join(out_dir, f"{task}_{i}.npy")
            np.save(path, patch)
            rows.append({"image_path": path, "label": label})
        return pd.DataFrame(rows)

    elif task == "prediction":
        for i in range(n_samples):
            seq = rng.normal(size=(seq_len, channels, patch_size, patch_size)).astype(np.float32)
            target = rng.normal(size=(horizon, 4)).astype(np.float32)
            seq_path = os.path.join(out_dir, f"seq_{i}.npy")
            tgt_path = os.path.join(out_dir, f"tgt_{i}.npy")
            np.save(seq_path, seq)
            np.save(tgt_path, target)
            rows.append({"sequence_path": seq_path, "target_path": tgt_path})
        return pd.DataFrame(rows)

    raise ValueError(f"Unknown task: {task}")
