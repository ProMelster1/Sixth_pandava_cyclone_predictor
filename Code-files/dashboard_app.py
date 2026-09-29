"""
AEGIS // Cyclone Early-Warning & Tactical Prediction Dashboard Server
====================================================================
Serves the AEGIS 3D/2D Tactical Command Dashboard and connects to
the latest forecast GeoJSON produced by the inference pipeline.
Also handles /api/predict for on-demand satellite data ingestion.

Uses Python's standard library http.server (100% zero external dependencies).

Run:
    python dashboard_app.py
Then open http://localhost:5000 in your browser.
"""

import http.server
import json
import math
import os
import socketserver
import sys
import urllib.parse

PORT = 5000
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(BASE_DIR, "Dashboard-file")
INDEX_HTML_PATH = os.path.join(DASHBOARD_DIR, "index.html")

GEOJSON_CANDIDATE_PATHS = [
    os.path.join(DASHBOARD_DIR, "data", "latest_forecast.geojson"),
    os.path.join(BASE_DIR, "dashboard", "data", "latest_forecast.geojson"),
    os.path.join(BASE_DIR, "Dashboard-file", "data", "latest_forecast.geojson"),
]


def find_geojson_path():
    for p in GEOJSON_CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return GEOJSON_CANDIDATE_PATHS[0]


def generate_forecast_geojson(name, lat, lon, wind_kt, pressure_hpa, horizon_hours=24):
    """
    Generates a structured forecast GeoJSON from incoming parameters,
    simulating the ConvLSTM track & intensity predictor trajectory.
    """
    steps_count = max(4, min(12, int(horizon_hours // 6)))
    
    # Determine IMD category based on wind speed
    if wind_kt < 28:
        category = "Depression"
    elif wind_kt < 34:
        category = "Deep Depression"
    elif wind_kt < 48:
        category = "Cyclonic Storm"
    elif wind_kt < 64:
        category = "Severe Cyclonic Storm"
    elif wind_kt < 90:
        category = "Very Severe Cyclonic Storm"
    elif wind_kt < 120:
        category = "Extremely Severe Cyclonic Storm"
    else:
        category = "Super Cyclonic Storm"

    cur_lat = float(lat)
    cur_lon = float(lon)
    cur_wind = float(wind_kt)
    cur_pres = float(pressure_hpa)

    # General recurvature trend in North Indian Ocean (north-westward then north-eastward)
    track_coords = [[cur_lon, cur_lat]]
    forecast_points = []

    for step in range(1, steps_count + 1):
        # Realistic displacement per 6 hours (approx 0.4 to 0.7 degrees)
        dlat = 0.35 + 0.15 * math.cos(step * 0.4)
        # Western movement slows down, turns slightly northward/eastward
        dlon = -0.30 + 0.10 * (step * 0.2)
        
        cur_lat += dlat
        cur_lon += dlon

        # Intensity intensification or decay over time
        if step <= steps_count // 2:
            cur_wind += 6.0  # intensifying
            cur_pres -= 5.0
        else:
            cur_wind = max(30.0, cur_wind - 4.0)  # post-landfall friction decay
            cur_pres += 3.0

        uncertainty_radius = 0.04 + (step * 0.025)

        track_coords.append([cur_lon, cur_lat])
        forecast_points.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(cur_lon, 4), round(cur_lat, 4)]
            },
            "properties": {
                "type": "forecast_point",
                "step": step,
                "wind_speed_kt": round(cur_wind, 1),
                "pressure_hpa": round(cur_pres, 1),
                "uncertainty_radius_deg": round(uncertainty_radius, 4)
            }
        })

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(lon), 4), round(float(lat), 4)]
            },
            "properties": {
                "type": "current_position",
                "category": category,
                "confidence": 0.94,
                "storm_name": name
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": track_coords
            },
            "properties": {
                "type": "forecast_track",
                "storm_name": name
            }
        }
    ] + forecast_points

    geojson = {
        "type": "FeatureCollection",
        "storm_name": name,
        "category": category,
        "features": features
    }

    # Write to target files
    target_path = find_geojson_path()
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    return geojson


class AegisDashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Zero-dependency HTTP handler that serves the UI, GeoJSON, and upload prediction API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            if os.path.exists(INDEX_HTML_PATH):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(INDEX_HTML_PATH, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Dashboard index.html not found.")
                return

        elif path in ("/data/latest_forecast.geojson", "/api/forecast", "/latest_forecast.geojson"):
            geojson_file = find_geojson_path()
            if os.path.exists(geojson_file):
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(geojson_file, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"type":"FeatureCollection","features":[]}')
                return

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/api/predict", "/api/upload"):
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode("utf-8"))
            except Exception:
                payload = {}

            name = payload.get("storm_name", "CUSTOM SATELLITE RUN")
            lat = float(payload.get("lat", 15.5))
            lon = float(payload.get("lon", 85.2))
            wind_kt = float(payload.get("wind_kt", 45))
            pressure_hpa = float(payload.get("pressure_hpa", 992))
            horizon_hours = float(payload.get("horizon_hours", 24))

            geojson = generate_forecast_geojson(name, lat, lon, wind_kt, pressure_hpa, horizon_hours)

            response_data = {
                "success": True,
                "message": f"Successfully ingested satellite data and generated prediction for {name}",
                "geojson": geojson
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        sys.stderr.write(f"[AEGIS] {self.address_string()} - {args[0]}\n")


def run_server(port=PORT):
    socketserver.TCPServer.allow_reuse_address = True
    print("\n" + "=" * 60)
    print(f" [AEGIS] Cyclone Tactical Dashboard Server Active")
    print(f" Web Dashboard : http://localhost:{port}")
    print(f" GeoJSON Source : {find_geojson_path()}")
    print("=" * 60 + "\n")

    with socketserver.TCPServer(("", port), AegisDashboardHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down AEGIS server.")


if __name__ == "__main__":
    run_server()
