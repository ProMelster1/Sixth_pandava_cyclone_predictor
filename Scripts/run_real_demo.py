"""
Real Demo Run
===============
Runs the full IDENTIFY -> CLASSIFY -> PREDICT -> uncertainty -> GeoJSON
pipeline on one real, named storm from your joined manifest (built by
build_incyde_manifest.py + join_ibtracs_track.py) instead of synthetic data.

Usage:
    python scripts/run_real_demo.py \
        --manifest incyde_manifest_with_track.csv \
        --storm PHAILIN \
        --seq_len 4 --horizon 4
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_loader import CYCLONE_CATEGORIES, load_image_any
from inference_pipeline import load_models, run_inference, to_geojson


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.manifest, parse_dates=["timestamp"])
    df = df[(df["name"] == args.storm.upper()) & (~df["is_augmented"])]
    df = df.dropna(subset=["LAT", "LON"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) < args.seq_len + 1:
        print(f"Only found {len(df)} real, track-matched frames for '{args.storm}'. "
              f"Need at least {args.seq_len + 1}. Try a different storm or lower --seq_len.")
        return

    window = df.iloc[-(args.seq_len + 1):-1]   # last seq_len frames as history
    current = df.iloc[-1]                       # most recent frame = "now"

    sequence = np.stack([load_image_any(p) for p in window["image_path"]])  # (T, 1, H, W)
    patch = load_image_any(current["image_path"])                            # (1, H, W)

    models = load_models(in_channels=1, num_classes=len(CYCLONE_CATEGORIES), horizon=args.horizon, device=device)

    result = run_inference(
        patch, sequence,
        lat=float(current["LAT"]), lon=float(current["LON"]),
        models=models, device=device, mc_samples=args.mc_samples,
    )

    print(json.dumps(result, indent=2))
    to_geojson(result, out_path="dashboard/data/latest_forecast.geojson")

    print(f"\nGround truth for reference — {args.storm} at {current['timestamp']}: "
          f"category={current.get('category')}, MSWS={current.get('msws_kts')} kt")
    print("Wrote dashboard/data/latest_forecast.geojson — run `python dashboard_app.py` to view it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Output of join_ibtracs_track.py")
    parser.add_argument("--storm", required=True, help="Storm name, e.g. PHAILIN")
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4, help="Must match --horizon used in train.py for the prediction task")
    parser.add_argument("--mc_samples", type=int, default=30)
    args = parser.parse_args()
    main(args)
