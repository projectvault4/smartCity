from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from utils.config import CONFIG, apply_city_config
from utils.forecast_service import build_comparison_payload, build_forecast_payload


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
APP_CONFIG = apply_city_config(CONFIG)


class ForecastRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/forecast":
            self._serve_forecast()
            return
        if parsed.path == "/api/comparison":
            self._serve_comparison()
            return
        if parsed.path == "/":
            self.path = "/index.html"
        if parsed.path == "/comparison":
            self.path = "/comparison.html"
        return super().do_GET()

    def _serve_forecast(self):
        try:
            payload = build_forecast_payload(APP_CONFIG)
            data = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
        except Exception as exc:
            data = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)

        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_comparison(self):
        try:
            payload = build_comparison_payload(APP_CONFIG)
            data = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
        except Exception as exc:
            data = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)

        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_server(host: str = "127.0.0.1", port: int = 8000, city: str | None = None) -> None:
    global APP_CONFIG
    APP_CONFIG = apply_city_config(APP_CONFIG, city)
    server = ThreadingHTTPServer((host, port), ForecastRequestHandler)
    print(f"Forecast website running at http://{host}:{port} for city={APP_CONFIG.city}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the forecast web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--city", default=None, help="Use city-specific data and outputs, e.g. delhi.")
    args = parser.parse_args()
    run_server(args.host, args.port, args.city)
