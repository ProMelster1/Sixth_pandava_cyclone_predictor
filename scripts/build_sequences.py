"""
Build Real Sequences for the Prediction Model
================================================
Turns the joined manifest (INCYDE images + real IBTrACS lat/lon/wind/pressure)
into sliding-window sequences: seq_len past frames -> horizon future steps of
(delta_lat, delta_lon, wind, pressure). Saves .npy files compatible with
CycloneSequenceDataset in dataset_loader.py.

Only uses non-augmented, track-matched rows (one real frame per timestamp) -
augmented duplicates would fake extra "time steps" that don't really exist.

Usage (from the repo root):
    python scripts/build_sequences.py \
        --manifest data/incyde_manifest_with_track.csv \
        --out_dir data/sequences \
        --output data/sequence_manifest.csv \
        --seq_len 4 --horizon 4
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
from config import to_data_relative
from dataset_loader import load_image_any


def build_sequences(manifest_path, out_dir, output_path, seq_len=4, horizon=4, min_split=None):
    df = pd.read_csv(manifest_path, parse_dates=["timestamp"])
    df = df[~df["is_augmented"]].dropna(subset=["LAT", "LON", "timestamp"])
    if min_split and "split" in df.columns:
        df = df[df["split"] == min_split]

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    window_len = seq_len + horizon

    for (name, year), group in df.groupby(["name", df["timestamp"].dt.year]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < window_len:
            continue

        for start in range(len(group) - window_len + 1):
            window = group.iloc[start:start + window_len]
            past = window.iloc[:seq_len]
            future = window.iloc[seq_len:]

            try:
                seq_images = np.stack([load_image_any(p) for p in past["image_path"]])
            except Exception as e:
                print(f"Skipping a window for {name} {year} — couldn't load an image: {e}")
                continue

            lats = future["LAT"].to_numpy(dtype=np.float32)
            lons = future["LON"].to_numpy(dtype=np.float32)
            wind_ibtracs = pd.to_numeric(future["INTENSITY_WIND_KT"], errors="coerce")
            wind_incyde = pd.to_numeric(future["msws_kts"], errors="coerce")

            winds = wind_ibtracs.fillna(wind_incyde).to_numpy(dtype=np.float32)
            press = pd.to_numeric(
                future["INTENSITY_PRES_MB"], errors="coerce"
            ).to_numpy(dtype=np.float32)

            last_lat, last_lon = past.iloc[-1]["LAT"], past.iloc[-1]["LON"]
            dlat = np.diff(np.concatenate([[last_lat], lats])).astype(np.float32)
            dlon = np.diff(np.concatenate([[last_lon], lons])).astype(np.float32)

            target = np.stack([dlat, dlon, winds, press], axis=1)  # (horizon, 4)
            if np.isnan(target).any():
                continue  # skip windows with any missing target value for a clean training signal

            idx = len(rows)
            seq_path = os.path.join(out_dir, f"seq_{name}_{year}_{idx}.npy")
            tgt_path = os.path.join(out_dir, f"tgt_{name}_{year}_{idx}.npy")
            np.save(seq_path, seq_images)
            np.save(tgt_path, target)

            rows.append({
                "sequence_path": to_data_relative(seq_path), "target_path": to_data_relative(tgt_path),
                "name": name, "year": year,
                "start_time": str(past.iloc[0]["timestamp"]),
                "split": past.iloc[-1].get("split", "train"),
            })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_path, index=False)
    print(f"Built {len(manifest)} sequences from {df['name'].nunique()} unique storms.")
    if len(manifest):
        print(manifest["split"].value_counts())
    else:
        print("No sequences built — check that enough consecutive real frames "
              "(non-augmented, track-matched, no missing target values) exist per storm.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Output of join_ibtracs_track.py")
    parser.add_argument("--out_dir", default="sequences")
    parser.add_argument("--output", default="sequence_manifest.csv")
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--min_split", default=None, help="Optionally restrict to one split value, e.g. train")
    args = parser.parse_args()
    build_sequences(args.manifest, args.out_dir, args.output, args.seq_len, args.horizon, args.min_split)
