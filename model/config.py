"""
Project paths
=============
Every script resolves files through these constants, so commands work from
any working directory on any machine.

Data files and manifests live in `data/`. Manifests store paths relative to
that folder (e.g. `images/08OCT2013_0830_1.jpg`), and `resolve_data_path`
turns them into real paths. Set CYCLONE_DATA_DIR to keep the (large) image
dataset somewhere else.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("CYCLONE_DATA_DIR", os.path.join(REPO_ROOT, "data"))
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")
FORECAST_GEOJSON = os.path.join(DASHBOARD_DIR, "data", "latest_forecast.geojson")


def resolve_data_path(path):
    """Absolute paths are used as-is; relative ones are looked up under DATA_DIR."""
    path = str(path).replace("\\", "/")
    return path if os.path.isabs(path) else os.path.join(DATA_DIR, path)


def to_data_relative(path):
    """Store paths inside DATA_DIR relative to it (portable); leave others unchanged."""
    abs_path = os.path.abspath(path)
    data_dir = os.path.abspath(DATA_DIR)
    if os.path.commonpath([abs_path, data_dir]) == data_dir:
        return os.path.relpath(abs_path, data_dir).replace("\\", "/")
    return str(path).replace("\\", "/")
