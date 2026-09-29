"""
Unified Training Script
=========================
Usage (from the repo root):
    python model/train.py --task classification --epochs 1 --synthetic
    python model/train.py --task classification --epochs 15 --in_channels 1 --manifest data/incyde_manifest.csv
    python model/train.py --task prediction --epochs 20 --in_channels 1 --manifest data/sequence_manifest.csv

Checkpoints are written to checkpoints/<task>_model.pt (synthetic runs go to
checkpoints/synthetic/ so they never overwrite real weights).

Use --synthetic to sanity-check the whole pipeline with generated dummy data
before wiring in real INSAT-3D/3DR + reanalysis data.
"""

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import (
    CycloneImageDataset,
    CycloneSequenceDataset,
    generate_synthetic_dataset,
)
from models import CycloneDetectionCNN, CycloneClassificationCNN, TrackIntensityPredictor
from config import CHECKPOINTS_DIR, DATA_DIR


def get_model(task, in_channels=4, num_classes=7, horizon=4):
    if task == "detection":
        return CycloneDetectionCNN(in_channels=in_channels)
    if task == "classification":
        return CycloneClassificationCNN(in_channels=in_channels, num_classes=num_classes)
    if task == "prediction":
        return TrackIntensityPredictor(in_channels=in_channels, horizon=horizon)
    raise ValueError(task)


def get_dataloader(task, manifest_df, batch_size=16):
    if task in ("detection", "classification"):
        ds = CycloneImageDataset(manifest_df, task=task)
    else:
        ds = CycloneSequenceDataset(manifest_df)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.synthetic or args.manifest is None:
        manifest = generate_synthetic_dataset(
            out_dir=os.path.join(DATA_DIR, f"synthetic_{args.task}"), task=args.task,
        )
    else:
        manifest = pd.read_csv(args.manifest)
        if "split" in manifest.columns:
            before = len(manifest)
            manifest = manifest[manifest["split"] == args.split].reset_index(drop=True)
            print(f"Filtered manifest to split='{args.split}': {len(manifest)}/{before} rows")

    loader = get_dataloader(args.task, manifest, batch_size=args.batch_size)
    model = get_model(args.task, in_channels=args.in_channels, num_classes=args.num_classes, horizon=args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.task in ("detection", "classification"):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.SmoothL1Loss()

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(loader.dataset)
        print(f"[{args.task}] epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f}")

    ckpt_dir = os.path.join(CHECKPOINTS_DIR, "synthetic") if (args.synthetic or args.manifest is None) else CHECKPOINTS_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.task}_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["detection", "classification", "prediction"], required=True)
    parser.add_argument("--manifest", default=None, help="CSV manifest path; omit to use --synthetic data")
    parser.add_argument("--synthetic", action="store_true", help="Train on generated dummy data (pipeline test)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--in_channels", type=int, default=4,
                         help="Use 1 for real INCYDE grayscale imagery, 4 for synthetic multi-band data")
    parser.add_argument("--num_classes", type=int, default=7, help="7 for real IMD categories (D..SuCS)")
    parser.add_argument("--split", default="train", help="Which 'split' column value to train on when the manifest has one")
    parser.add_argument("--horizon", type=int, default=4, help="Prediction-task forecast steps; must match at inference time")
    args = parser.parse_args()
    train(args)
  