"""
End-to-End Inference Pipeline
================================
Mirrors the workflow diagram:

  raw multi-source data -> preprocessing -> AI/ML engine
        -> IDENTIFY (detection)
        -> CLASSIFY (pattern/stage)                    [if a cyclone is found]
        -> PREDICT  (track + intensity)                [if a cyclone is found]
        -> uncertainty estimation
        -> GeoJSON output for the GIS/web dashboard
"""

import json
import os

import numpy as np
import torch

from config import CHECKPOINTS_DIR, FORECAST_GEOJSON
from dataset_loader import CYCLONE_CATEGORIES
from models import CycloneDetectionCNN, CycloneClassificationCNN, TrackIntensityPredictor
from uncertainty import predict_with_uncertainty


def load_models(checkpoint_dir=CHECKPOINTS_DIR, in_channels=1,
                num_classes=len(CYCLONE_CATEGORIES), horizon=4,
                device="cpu"):

    # Step 6: No real detector is trained because the INCYDE dataset
    # contains no negative/no-cyclone samples.
    detector = None

    classifier = CycloneClassificationCNN(
        in_channels=in_channels,
        num_classes=num_classes
    ).to(device)

    predictor = TrackIntensityPredictor(
        in_channels=in_channels,
        horizon=horizon
    ).to(device)

    # Load the trained classification model
    classification_path = os.path.join(
        checkpoint_dir, "classification_model.pt"
    )
    if os.path.exists(classification_path):
        classifier.load_state_dict(
            torch.load(classification_path, map_location=device)
        )

    # Load the trained prediction model
    prediction_path = os.path.join(
        checkpoint_dir, "prediction_model.pt"
    )
    if os.path.exists(prediction_path):
        predictor.load_state_dict(
            torch.load(prediction_path, map_location=device)
        )

    classifier.eval()
    predictor.eval()

    return detector, classifier, predictor

def run_inference(patch, sequence, lat, lon, models, device="cpu",
                   detection_threshold=0.5, mc_samples=30):
    """
    patch    : np.ndarray (C, H, W)  - current preprocessed frame for detect/classify
    sequence : np.ndarray (T, C, H, W) - recent frame history for track/intensity prediction
    lat, lon : current best-estimate cyclone center (for track propagation)
    """
    detector, classifier, predictor = models

    x_patch = torch.from_numpy(patch).unsqueeze(0).float().to(device)
    if detector is None:
        # No trained detector (INCYDE has no cyclone-free frames): every
        # INCYDE frame is a tracked storm, so go straight to classification.
        result = {"detection": {"cyclone_present": True, "confidence": None, "uncertainty_entropy": None}}
        cyclone_present = True
    else:
        det_probs, det_entropy = predict_with_uncertainty(detector, x_patch, n_samples=mc_samples, task="detection")
        cyclone_present = bool(det_probs[0, 1] > detection_threshold)
        result = {
            "detection": {
                "cyclone_present": cyclone_present,
                "confidence": float(det_probs[0, 1]),
                "uncertainty_entropy": float(det_entropy[0]),
            }
        }

    if not cyclone_present:
        return result

    class_probs, class_entropy = predict_with_uncertainty(classifier, x_patch, n_samples=mc_samples, task="classification")
    pred_class = int(torch.argmax(class_probs, dim=-1)[0])
    result["classification"] = {
        "category": CYCLONE_CATEGORIES[pred_class],
        "confidence": float(class_probs[0, pred_class]),
        "uncertainty_entropy": float(class_entropy[0]),
    }

    x_seq = torch.from_numpy(sequence).unsqueeze(0).float().to(device)
    mean_forecast, std_forecast = predict_with_uncertainty(predictor, x_seq, n_samples=mc_samples, task="prediction")
    mean_forecast = mean_forecast[0].cpu().numpy()   # (horizon, 4)
    std_forecast = std_forecast[0].cpu().numpy()

    track = []
    cur_lat, cur_lon = lat, lon
    for step, (dlat, dlon, wind, pressure) in enumerate(mean_forecast):
        cur_lat += float(dlat)
        cur_lon += float(dlon)
        track.append({
            "step": step + 1,
            "lat": cur_lat,
            "lon": cur_lon,
            "wind_speed_kt": float(wind),
            "pressure_hpa": float(pressure),
            "lat_std": float(std_forecast[step, 0]),
            "lon_std": float(std_forecast[step, 1]),
            "wind_std": float(std_forecast[step, 2]),
            "pressure_std": float(std_forecast[step, 3]),
        })
    result["prediction"] = {"origin": {"lat": lat, "lon": lon}, "forecast_track": track}
    return result


def to_geojson(result, out_path=FORECAST_GEOJSON):
    """Converts a single inference result into a GeoJSON FeatureCollection the dashboard can render."""
    features = []

    if result.get("detection", {}).get("cyclone_present") and "prediction" in result:
        origin = result["prediction"]["origin"]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [origin["lon"], origin["lat"]]},
            "properties": {
                "type": "current_position",
                "category": result.get("classification", {}).get("category"),
                "confidence": result.get("classification", {}).get("confidence"),
            },
        })

        coords = [[origin["lon"], origin["lat"]]] + [
            [p["lon"], p["lat"]] for p in result["prediction"]["forecast_track"]
        ]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"type": "forecast_track"},
        })

        for p in result["prediction"]["forecast_track"]:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                "properties": {
                    "type": "forecast_point",
                    "step": p["step"],
                    "wind_speed_kt": p["wind_speed_kt"],
                    "pressure_hpa": p["pressure_hpa"],
                    "uncertainty_radius_deg": max(p["lat_std"], p["lon_std"]),
                },
            })

    geojson = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    return geojson


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device=device)

    # --- Smoke test on random single-channel frames (real INCYDE frames are 1 x 128 x 128) ---
    rng = np.random.default_rng(0)
    dummy_patch = rng.normal(size=(1, 128, 128)).astype(np.float32)
    dummy_sequence = rng.normal(size=(4, 1, 128, 128)).astype(np.float32)
    # ---------------------------------------------------------------

    result = run_inference(
        dummy_patch, dummy_sequence, lat=15.5, lon=85.2, models=models, device=device,
    )
    to_geojson(result)
    print(json.dumps(result, indent=2))
