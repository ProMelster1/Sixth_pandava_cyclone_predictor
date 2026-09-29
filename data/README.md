# Data

Paths inside the manifests are **relative to this folder** (for example `images/08OCT2013_0830_1.jpg`), so they work on any machine. To keep the images elsewhere, set `CYCLONE_DATA_DIR` to that folder.

## Included in the repo

| File | What it is |
|---|---|
| `Cropped_cyclone_labels.csv` | INCYDE labels: storm name, MSWS (kt), IMD category, filename, train/test split |
| `ibtracs_clean.csv` | North Indian Ocean best tracks from IBTrACS (lat/lon, wind, pressure every 3–6 h) |
| `incyde_manifest.csv` | Output of `scripts/build_incyde_manifest.py`: one row per image, with class index and timestamp |
| `incyde_manifest_with_track.csv` | Output of `scripts/join_ibtracs_track.py`: the manifest joined to IBTrACS positions and intensity |
| `sequence_manifest.csv` | Output of `scripts/build_sequences.py`: 8,511 training windows (4 past frames → 4 future steps) |

## Not included — download or regenerate

| Folder | How to get it |
|---|---|
| `images/` | The INCYDE cropped INSAT-3D IR images (`Cropped_cyclone_images/`, ~103k JPGs). Download the INCYDE dataset and copy the images **directly** into `data/images/`. |
| `sequences/` | Rebuilt from the images: `python scripts/build_sequences.py --manifest data/incyde_manifest_with_track.csv --out_dir data/sequences --output data/sequence_manifest.csv` |

Both folders are git-ignored because of their size.
