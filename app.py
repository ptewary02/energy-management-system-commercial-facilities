"""
app.py - Flask backend for the AI-powered energy management prototype
------------------------------------------------------------------------
Two endpoints the ESP32 talks to:

  POST /readings   ESP32 sends its current 4-zone sensor readings here.
                    Body: {"zone1": 2.3, "zone2": 5.1, "zone3": 1.8,
                            "zone4": 6.0, "renewable": 3.2}

  GET  /commands    ESP32 polls this to find out which zones should be
                     ON or OFF right now.
                     Response: {"zone1": true, "zone2": false,
                                "zone3": true, "zone4": true, ...}

Pipeline behind /commands:
  1. Load recent data (real logged readings if we have enough, else fall
     back to the synthetic dataset so the system is demoable from day one).
  2. Forecast the next 24h per zone with forecast_engine.py.
  3. Build a renewable forecast from the historical hour-of-day average
     (a simple stand-in - swap in a real solar API/LDR-trained model later).
  4. Run optimizer.py to get the recommended controllable allocation per hour.
  5. Map the CURRENT hour's target allocation down to individual zone
     on/off decisions (relays are binary, the optimizer's output is
     continuous - see the mapping function below).

Run with: python app.py
Find your machine's local IP (not localhost) with `ipconfig`/`ifconfig`
so the ESP32 can reach this over WiFi, e.g. http://192.168.1.42:5000
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from forecast_engine import train_zone_model, forecast_next_24h
from optimizer import optimize_day

app = Flask(__name__)
CORS(app)  # allows the local dashboard (opened as a file or on another port) to fetch this API

READINGS_LOG_PATH = "readings_log.csv"
SYNTHETIC_FALLBACK_PATH = "synthetic_energy_dataset.csv"
MIN_REAL_ROWS_TO_USE = 24 * 7  # need at least a week of real data before trusting it alone
MODEL_CACHE_TTL_SECONDS = 60 * 60  # retrain at most once per hour

ZONE_COLUMNS = {
    "zone1": "Zone1_Lighting_kW",
    "zone2": "Zone2_HVAC_kW",
    "zone3": "Zone3_PlugLoads_kW",
    "zone4": "Zone4_Critical_kW",
}
CONTROLLABLE_ZONES = ["zone1", "zone2", "zone3"]  # zone4 is priority, never shifted

_model_cache = {"trained_at": 0, "fits": {}, "history": None}


def tariff_for_hour(hour: int) -> float:
    """Same time-of-use tariff schedule used to generate the synthetic dataset."""
    is_peak = (6 <= hour < 10) or (18 <= hour < 22)
    return 9.5 if is_peak else 5.5


def load_history() -> pd.DataFrame:
    """Loads real logged readings if we have enough, otherwise falls back
    to the synthetic dataset so /commands works even before real data
    has accumulated."""
    if os.path.exists(READINGS_LOG_PATH):
        real = pd.read_csv(READINGS_LOG_PATH, parse_dates=["Datetime"])
        if len(real) >= MIN_REAL_ROWS_TO_USE:
            return real.set_index("Datetime")

    synthetic = pd.read_csv(SYNTHETIC_FALLBACK_PATH, parse_dates=["Datetime"])
    return synthetic.set_index("Datetime")


def get_or_train_models():
    """Returns cached SARIMAX fits per zone, retraining if the cache is
    stale or empty. Retraining on every request would be far too slow -
    SARIMA fitting takes real time, so we only redo it hourly."""
    now = time.time()
    if now - _model_cache["trained_at"] < MODEL_CACHE_TTL_SECONDS and _model_cache["fits"]:
        return _model_cache["fits"], _model_cache["history"]

    history = load_history()
    history.index.freq = pd.infer_freq(history.index) or "h"

    fits = {}
    for zone_key, col in ZONE_COLUMNS.items():
        training_slice = history[col].iloc[-24 * 30:]  # last 30 days, same as the notebook
        fits[zone_key] = train_zone_model(training_slice)

    _model_cache["trained_at"] = now
    _model_cache["fits"] = fits
    _model_cache["history"] = history
    return fits, history


def renewable_forecast_next_24h(history: pd.DataFrame, last_timestamp: pd.Timestamp) -> np.ndarray:
    """Simple stand-in renewable forecast: average renewable supply by hour-of-day
    from history. Replace with a real solar-forecast API or a trained model on
    your LDR readings once you have enough of that data logged."""
    hourly_avg = history.groupby(history.index.hour)["Renewable_Supply_kW"].mean()
    future_hours = [(last_timestamp.hour + i + 1) % 24 for i in range(24)]
    return hourly_avg.reindex(future_hours).values


def map_allocation_to_relays(target_kw: float, zone_forecast_kw: dict) -> dict:
    """Translates the optimizer's continuous target allocation (kW) for the
    CURRENT hour into binary on/off decisions per controllable zone.

    Simple greedy rule: turn on the largest forecasted zones first until
    their combined power reaches the target. This is a starting heuristic -
    a more refined version could weight by shiftability or priority instead
    of raw size.
    """
    relay_state = {"zone4": True}  # priority zone is always on

    zones_sorted = sorted(CONTROLLABLE_ZONES, key=lambda z: zone_forecast_kw[z], reverse=True)
    running_total = 0.0
    for zone in zones_sorted:
        if running_total < target_kw:
            relay_state[zone] = True
            running_total += zone_forecast_kw[zone]
        else:
            relay_state[zone] = False

    return relay_state


@app.route("/readings", methods=["POST"])
def post_readings():
    data = request.get_json(force=True)
    required = ["zone1", "zone2", "zone3", "zone4"]
    if not all(k in data for k in required):
        return jsonify({"error": f"expected keys {required}"}), 400

    row = {
        "Datetime": datetime.now().replace(microsecond=0),
        "Zone1_Lighting_kW": data["zone1"],
        "Zone2_HVAC_kW": data["zone2"],
        "Zone3_PlugLoads_kW": data["zone3"],
        "Zone4_Critical_kW": data["zone4"],
        "Renewable_Supply_kW": data.get("renewable", 0.0),
    }

    file_exists = os.path.exists(READINGS_LOG_PATH)
    pd.DataFrame([row]).to_csv(READINGS_LOG_PATH, mode="a", header=not file_exists, index=False)

    return jsonify({"status": "logged", "row": {k: str(v) for k, v in row.items()}})


@app.route("/commands", methods=["GET"])
def get_commands():
    fits, history = get_or_train_models()
    last_timestamp = history.index[-1]

    zone_forecasts = {}
    for zone_key in ZONE_COLUMNS:
        forecast_df = forecast_next_24h(fits[zone_key], last_timestamp)
        zone_forecasts[zone_key] = forecast_df["predicted_kW"].clip(lower=0).values

    priority = zone_forecasts["zone4"]
    controllable = sum(zone_forecasts[z] for z in CONTROLLABLE_ZONES)
    renewable = renewable_forecast_next_24h(history, last_timestamp)
    tariff = np.array([tariff_for_hour((last_timestamp.hour + i + 1) % 24) for i in range(24)])

    result = optimize_day(priority, controllable, renewable, tariff)

    current_hour_target = result["recommended_allocation_kw"][0]
    current_zone_forecast_kw = {z: float(zone_forecasts[z][0]) for z in CONTROLLABLE_ZONES}
    relay_state = map_allocation_to_relays(current_hour_target, current_zone_forecast_kw)

    return jsonify({
        "relay_state": relay_state,
        "current_hour_target_kw": round(current_hour_target, 2),
        "metrics": {k: v for k, v in result.items() if not isinstance(v, list)},
        "curves": {
            "baseline_grid_draw": result["baseline_grid_draw"],
            "optimized_grid_draw": result["optimized_grid_draw"],
            "recommended_allocation_kw": result["recommended_allocation_kw"],
            "capacity_target_kw": result["capacity_target_kw"],
        },
        "generated_at": last_timestamp.isoformat(),
    })


@app.route("/latest", methods=["GET"])
def get_latest():
    if not os.path.exists(READINGS_LOG_PATH):
        return jsonify({"error": "no readings logged yet - waiting for the ESP32 to post"}), 404

    df = pd.read_csv(READINGS_LOG_PATH, parse_dates=["Datetime"])
    if df.empty:
        return jsonify({"error": "no readings logged yet - waiting for the ESP32 to post"}), 404

    recent = df.tail(60)  # roughly last 10 minutes at a 10s post interval
    latest_row = recent.iloc[-1]

    return jsonify({
        "latest": {
            "timestamp": latest_row["Datetime"].isoformat(),
            "zone1_kw": round(float(latest_row["Zone1_Lighting_kW"]), 3),
            "zone2_kw": round(float(latest_row["Zone2_HVAC_kW"]), 3),
            "zone3_kw": round(float(latest_row["Zone3_PlugLoads_kW"]), 3),
            "zone4_kw": round(float(latest_row["Zone4_Critical_kW"]), 3),
            "renewable_kw": round(float(latest_row.get("Renewable_Supply_kW", 0.0)), 3),
        },
        "history": {
            "timestamps": recent["Datetime"].dt.strftime("%H:%M:%S").tolist(),
            "zone1_kw": recent["Zone1_Lighting_kW"].round(3).tolist(),
            "zone2_kw": recent["Zone2_HVAC_kW"].round(3).tolist(),
            "zone3_kw": recent["Zone3_PlugLoads_kW"].round(3).tolist(),
            "zone4_kw": recent["Zone4_Critical_kW"].round(3).tolist(),
        }
    })


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Energy management backend is running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)