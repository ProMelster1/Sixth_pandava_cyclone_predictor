"""
Build INCYDE Manifest
=======================
Turns the raw INCYDE Cropped_cyclone_labels.csv + Cropped_cyclone_images/
folder into a clean manifest for training: verifies every image actually
exists on disk, maps IMD category letters to class indices, parses the
real timestamp out of each filename, and respects the dataset's own
Train/Test split (no leakage).

Confirmed real label columns:
    Sequence, Name, MSWS kts, Category, filename, Set
(an extra pandas index column, e.g. "Unnamed: 0", is dropped automatically
if present)

The `filename` column already contains the COMPLETE real filename,
extension and augmentation suffix included, e.g.:
    08OCT2013_0830_1.jpg              (base image)
    08OCT2013_0830_1_flip_h.jpg       (augmented variant)
    08OCT2013_0830_1_flip_v.jpg
    08OCT2013_0830_1_rot_90.jpg
    08OCT2013_0830_1_rot_180.jpg
    08OCT2013_0830_1_rot_270.jpg
So each row maps to exactly one file - no guessing needed, just a direct
os.path.join(images_dir, filename) lookup.

Usage (from the repo root):
    python scripts/build_incyde_manifest.py \
        --labels data/Cropped_cyclone_labels.csv \
        --images_dir data/images \
        --output data/incyde_manifest.csv \
        --include_augmented

Image paths inside data/ are stored relative to it, so the manifest stays portable.
"""

import argparse
import os
import re

import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
from config import to_data_relative

# IMD cyclone category scale actually present in this dataset (no "Low
# Pressure Area" here since these are all already-named, tracked storms).
CATEGORY_MAP = {
    "D": 0,     # Depression
    "DD": 1,    # Deep Depression
    "CS": 2,    # Cyclonic Storm
    "SCS": 3,   # Severe Cyclonic Storm
    "VSCS": 4,  # Very Severe Cyclonic Storm
    "ESCS": 5,  # Extremely Severe Cyclonic Storm
    "SUCS": 6,  # Super Cyclonic Storm
}
CATEGORY_NAMES = ["Depression", "Deep Depression", "Cyclonic Storm",
                   "Severe Cyclonic Storm", "Very Severe Cyclonic Storm",
                   "Extremely Severe Cyclonic Storm", "Super Cyclonic Storm"]

AUGMENT_MARKERS = ("_flip_h", "_flip_v", "_rot_90", "_rot_180", "_rot_270")

# e.g. "08OCT2013_0830" -> day=08, mon=OCT, year=2013, time=0830
FNAME_RE = re.compile(r"(\d{2})([A-Z]{3})(\d{4})_(\d{4})")
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def parse_timestamp(filename_str):
    """Parses '08OCT2013_0830...' -> pandas Timestamp. Returns NaT if it doesn't match."""
    m = FNAME_RE.search(str(filename_str).upper())
    if not m:
        return pd.NaT
    day, mon, year, hhmm = m.groups()
    month = MONTHS.get(mon)
    if month is None:
        return pd.NaT
    hour, minute = int(hhmm[:2]), int(hhmm[2:])
    try:
        return pd.Timestamp(year=int(year), month=month, day=int(day), hour=hour, minute=minute)
    except ValueError:
        return pd.NaT


def is_augmented(filename_str):
    return any(marker in str(filename_str) for marker in AUGMENT_MARKERS)


def build_manifest(labels_path, images_dir, output_path, include_augmented=False):
    df = pd.read_csv(labels_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # drop stray pandas index column

    name_col = next(c for c in df.columns if c.lower() == "name")
    wind_col = next(c for c in df.columns if "msws" in c.lower())
    cat_col = next(c for c in df.columns if c.lower() == "category")
    fname_col = next(c for c in df.columns if "file" in c.lower())
    set_col = next((c for c in df.columns if c.lower() == "set"), None)

    rows = []
    unmatched = []

    for _, row in df.iterrows():
        fname = str(row[fname_col]).strip()
        augmented = is_augmented(fname)
        split = str(row[set_col]).strip().lower() if set_col else "train"

        # Only keep augmented variants for Train rows, and only if requested -
        # keeps Test/Val strictly on real, non-augmented images.
        if augmented and (not include_augmented or split != "train"):
            continue

        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            unmatched.append(row.to_dict())
            continue

        category = str(row[cat_col]).strip().upper()
        if category not in CATEGORY_MAP:
            unmatched.append(row.to_dict())
            continue

        rows.append({
            "image_path": to_data_relative(img_path),
            "label": CATEGORY_MAP[category],
            "category": category,
            "msws_kts": row[wind_col],
            "name": str(row[name_col]).strip().upper(),
            "timestamp": parse_timestamp(fname),
            "split": split,
            "is_augmented": augmented,
        })

    manifest = pd.DataFrame(rows)

    print(f"Matched {len(manifest)} image rows "
          f"({int(manifest['is_augmented'].sum()) if len(manifest) else 0} augmented) "
          f"from {len(df)} label rows. {len(unmatched)} label rows had no matching image, "
          f"unknown category, or were filtered augmented rows.")
    if unmatched:
        pd.DataFrame(unmatched).to_csv(output_path.replace(".csv", "_unmatched.csv"), index=False)
        print(f"Unmatched rows saved to {output_path.replace('.csv', '_unmatched.csv')} for a quick manual check.")

    if len(manifest):
        print("\nClass distribution:")
        print(manifest["category"].value_counts())
        print("\nSplit distribution:")
        print(manifest["split"].value_counts())
        print(f"\nUnique storms: {manifest['name'].nunique()}")
        n_missing_ts = manifest["timestamp"].isna().sum()
        if n_missing_ts:
            print(f"Warning: {n_missing_ts} rows had a filename timestamp that didn't parse - "
                  f"these won't be usable for track/sequence building.")

    manifest.to_csv(output_path, index=False)
    print(f"\nSaved manifest to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output", default="incyde_manifest.csv")
    parser.add_argument("--include_augmented", action="store_true",
                         help="Include the pre-made flip/rotate variant rows for TRAIN only (free data augmentation, no test leakage)")
    args = parser.parse_args()
    build_manifest(args.labels, args.images_dir, args.output, args.include_augmented)
