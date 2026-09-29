"""
The Sixth Pandava — local dashboard server
==========================================
Serves dashboard/ and the latest forecast GeoJSON. Standard library only.

Run from the repo root:
    python serve.py            # http://localhost:5000
    python serve.py --port 8080

The dashboard is a static site, so any static host (e.g. GitHub Pages) works too.
A server is needed locally only because browsers block loading the forecast
file from file:// URLs. POST /api/predict generates a demo forecast from
parameters and writes it to dashboard/data/latest_forecast.geojson.
"""

import argparse
import http.server
import json
import math
import os
import socketserver
import sys
import urllib.parse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")
GEOJSON_PATH = os.path.join(DASHBOARD_DIR, "data", "latest_forecast.geojson")


def imd_category(wind_kt):
    if wind_kt < 28:
        return "Depression"
    if wind_kt < 34:
        return "Deep Depression"
    if wind_kt < 48:
        return "Cyclonic Storm"
    if wind_kt < 64:
        return "Severe Cyclonic Storm"
    if wind_kt < 90:
        return "Very Severe Cyclonic Storm"
    if wind_kt < 120:
        return "Extremely Severe Cyclonic Storm"
    return "Super Cyclonic Storm"


def generate_forecast_geojson(name, lat, lon, wind_kt, pressure_hpa, horizon_hours=24):
    """Demo forecast: a typical North Indian Ocean recurving track with
    intensification then post-landfall decay (not the trained model)."""
    steps_count = max(4, min(12, int(horizon_hours // 6)))
    category = imd_category(wind_kt)
    cur_lat, cur_lon, cur_wind, cur_pres = float(lat), float(lon), float(wind_kt), float(pressure_hpa)

    track_coords = [[cur_lon, cur_lat]]
    forecast_points = []
    for step in range(1, steps_count + 1):
        cur_lat += 0.35 + 0.15 * math.cos(step * 0.4)
        cur_lon += -0.30 + 0.10 * (step * 0.2)
        if step <= steps_count // 2:
            cur_wind += 6.0
            cur_pres -= 5.0
        else:
            cur_wind = max(30.0, cur_wind - 4.0)
            cur_pres += 3.0
        track_coords.append([cur_lon, cur_lat])
        forecast_points.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(cur_lon, 4), round(cur_lat, 4)]},
            "properties": {
                "type": "forecast_point",
                "step": step,
                "wind_speed_kt": round(cur_wind, 1),
                "pressure_hpa": round(cur_pres, 1),
                "uncertainty_radius_deg": round(0.04 + step * 0.025, 4),
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "storm_name": name,
        "category": category,
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(float(lon), 4), round(float(lat), 4)]},
                "properties": {"type": "current_position", "category": category, "confidence": 0.94, "storm_name": name},
            },
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": track_coords},
                "properties": {"type": "forecast_track", "storm_name": name},
            },
        ] + forecast_points,
    }
    os.makedirs(os.path.dirname(GEOJSON_PATH), exist_ok=True)
    with open(GEOJSON_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    return geojson


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def _send_json(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/api/forecast", "/latest_forecast.geojson"):
            if os.path.exists(GEOJSON_PATH):
                with open(GEOJSON_PATH, "rb") as f:
                    return self._send_json(200, f.read())
            return self._send_json(404, b'{"type":"FeatureCollection","features":[]}')
        return super().do_GET()

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path not in ("/api/predict", "/api/upload"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = {}
        name = payload.get("storm_name", "CUSTOM SATELLITE RUN")
        geojson = generate_forecast_geojson(
            name,
            float(payload.get("lat", 15.5)),
            float(payload.get("lon", 85.2)),
            float(payload.get("wind_kt", 45)),
            float(payload.get("pressure_hpa", 992)),
            float(payload.get("horizon_hours", 24)),
        )
        body = {"success": True, "message": f"Generated forecast for {name}", "geojson": geojson}
        self._send_json(200, json.dumps(body).encode("utf-8"))

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        sys.stderr.write(f"[sixth-pandava] {self.address_string()} - {args[0]}\n")


def main():
    parser = argparse.ArgumentParser(description="Serve The Sixth Pandava dashboard")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    args = parser.parse_args()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), DashboardHandler) as httpd:
        print(f"The Sixth Pandava dashboard: http://localhost:{args.port}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
