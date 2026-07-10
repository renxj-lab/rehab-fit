import asyncio
import struct
import time
import os
import pickle
import joblib
import threading
from collections import deque

import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
from bleak import BleakClient


# =====================================================
# FLASK APP
# =====================================================
app = Flask(__name__)
CORS(app)


# =====================================================
# WT901BLE SENSOR SETTINGS
# =====================================================
ADDRESS = "E1:05:3D:86:3A:03"
NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"


# =====================================================
# MODEL FILES
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "wt901_live_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "wt901_live_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "wt901_live_features.pkl")


# =====================================================
# LIVE GLOBAL DATA
# =====================================================
WINDOW_SIZE = 100

latest_reading = {
    "time": time.time(),
    "ax": 0,
    "ay": 0,
    "az": 0,
    "wx": 0,
    "wy": 0,
    "wz": 0,
    "angle_x": 0,
    "angle_y": 0,
    "angle_z": 0,
}

live_window = deque(maxlen=WINDOW_SIZE)

sensor_running = False
sensor_connected = False
sensor_status = "Sensor not started"
sensor_thread = None
stop_event = threading.Event()

latest_prediction = {
    "exercise": "Waiting",
    "confidence": 0,
    "feedback": "Start the WIT sensor to begin.",
}


# =====================================================
# LOAD MODEL
# =====================================================
def load_file(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None

    try:
        return joblib.load(path)
    except Exception:
        pass

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Could not load: {path}")
        print(e)
        return None


model = load_file(MODEL_PATH)
scaler = load_file(SCALER_PATH)
feature_names = load_file(FEATURES_PATH)

print("==========================================")
print("REHAB FIT BACKEND")
print("==========================================")
print("Model loaded:", model is not None)
print("Scaler loaded:", scaler is not None)
print("Features loaded:", feature_names is not None)

if feature_names is not None:
    print("Number of features:", len(feature_names))

print("==========================================")


# =====================================================
# WT901BLE PACKET PARSER
# =====================================================
def parse_wt901_packet(packet):
    """
    WT901BLE BLE packet format:
    20 bytes:
    55 61 ax ay az wx wy wz angle_x angle_y angle_z
    """

    if len(packet) != 20:
        return None

    if packet[0] != 0x55 or packet[1] != 0x61:
        return None

    try:
        values = struct.unpack("<hhhhhhhhh", packet[2:20])
    except Exception:
        return None

    (
        ax_raw,
        ay_raw,
        az_raw,
        wx_raw,
        wy_raw,
        wz_raw,
        roll_raw,
        pitch_raw,
        yaw_raw,
    ) = values

    ax = ax_raw / 32768.0 * 16.0
    ay = ay_raw / 32768.0 * 16.0
    az = az_raw / 32768.0 * 16.0

    wx = wx_raw / 32768.0 * 2000.0
    wy = wy_raw / 32768.0 * 2000.0
    wz = wz_raw / 32768.0 * 2000.0

    angle_x = roll_raw / 32768.0 * 180.0
    angle_y = pitch_raw / 32768.0 * 180.0
    angle_z = yaw_raw / 32768.0 * 180.0

    return {
        "time": time.time(),
        "ax": float(ax),
        "ay": float(ay),
        "az": float(az),
        "wx": float(wx),
        "wy": float(wy),
        "wz": float(wz),
        "angle_x": float(angle_x),
        "angle_y": float(angle_y),
        "angle_z": float(angle_z),
    }


# =====================================================
# FEATURE EXTRACTION
# =====================================================
def safe_skew(values):
    values = np.array(values, dtype=float)

    if len(values) < 2:
        return 0.0

    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return 0.0

    return float(np.mean(((values - mean) / std) ** 3))


def safe_kurtosis(values):
    values = np.array(values, dtype=float)

    if len(values) < 2:
        return 0.0

    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return 0.0

    return float(np.mean(((values - mean) / std) ** 4) - 3)


def zero_crossings(values):
    values = np.array(values, dtype=float)

    if len(values) < 2:
        return 0.0

    return float(np.sum(np.diff(np.sign(values)) != 0))


def extract_all_features(window):
    cols = [
        "ax",
        "ay",
        "az",
        "wx",
        "wy",
        "wz",
        "angle_x",
        "angle_y",
        "angle_z",
    ]

    feature_dict = {}

    for col in cols:
        values = np.array([row.get(col, 0.0) for row in window], dtype=float)

        feature_dict[f"{col}_mean"] = float(np.mean(values))
        feature_dict[f"{col}_std"] = float(np.std(values))
        feature_dict[f"{col}_min"] = float(np.min(values))
        feature_dict[f"{col}_max"] = float(np.max(values))
        feature_dict[f"{col}_range"] = float(np.max(values) - np.min(values))
        feature_dict[f"{col}_rms"] = float(np.sqrt(np.mean(values ** 2)))
        feature_dict[f"{col}_var"] = float(np.var(values))
        feature_dict[f"{col}_median"] = float(np.median(values))
        feature_dict[f"{col}_energy"] = float(np.sum(values ** 2))
        feature_dict[f"{col}_skew"] = safe_skew(values)
        feature_dict[f"{col}_kurtosis"] = safe_kurtosis(values)
        feature_dict[f"{col}_zero_crossings"] = zero_crossings(values)

    ax = np.array([row.get("ax", 0.0) for row in window], dtype=float)
    ay = np.array([row.get("ay", 0.0) for row in window], dtype=float)
    az = np.array([row.get("az", 0.0) for row in window], dtype=float)

    wx = np.array([row.get("wx", 0.0) for row in window], dtype=float)
    wy = np.array([row.get("wy", 0.0) for row in window], dtype=float)
    wz = np.array([row.get("wz", 0.0) for row in window], dtype=float)

    acc_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    gyro_mag = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)

    for name, values in [("acc_mag", acc_mag), ("gyro_mag", gyro_mag)]:
        feature_dict[f"{name}_mean"] = float(np.mean(values))
        feature_dict[f"{name}_std"] = float(np.std(values))
        feature_dict[f"{name}_min"] = float(np.min(values))
        feature_dict[f"{name}_max"] = float(np.max(values))
        feature_dict[f"{name}_range"] = float(np.max(values) - np.min(values))
        feature_dict[f"{name}_rms"] = float(np.sqrt(np.mean(values ** 2)))
        feature_dict[f"{name}_var"] = float(np.var(values))
        feature_dict[f"{name}_median"] = float(np.median(values))
        feature_dict[f"{name}_energy"] = float(np.sum(values ** 2))
        feature_dict[f"{name}_skew"] = safe_skew(values)
        feature_dict[f"{name}_kurtosis"] = safe_kurtosis(values)
        feature_dict[f"{name}_zero_crossings"] = zero_crossings(values)

    return feature_dict


def make_model_input(window):
    if feature_names is None:
        return None, 0

    feature_dict = extract_all_features(window)

    row = []
    missing = 0

    for name in feature_names:
        if name not in feature_dict:
            missing += 1

        row.append(feature_dict.get(name, 0.0))

    X = np.array(row, dtype=float).reshape(1, -1)

    return X, missing


# =====================================================
# ABD / FEL CORRECTION BASED ON YOUR AXIS CHECK
# =====================================================
def correct_abd_fel_prediction(prediction, window):
    """
    ABD and FEL are both arm elevation movements.

    From your recorded WIT data:

    ABD:
    angle_x_range ≈ 117.56
    angle_y_range ≈ 47.03
    wy_range      ≈ 320.50
    wz_range      ≈ 362.96

    FEL:
    angle_x_range ≈ 130.20
    angle_y_range ≈ 38.45
    wy_range      ≈ 334.28
    wz_range      ≈ 210.02

    Meaning:
    - angle_x is high for both, so it is not enough alone.
    - ABD has higher angle_y range.
    - ABD has much higher wz range.
    - FEL has lower wz range and usually higher angle_x.
    """

    prediction = str(prediction)

    if prediction not in ["ABD", "FEL"]:
        return prediction

    angle_x = np.array([row.get("angle_x", 0.0) for row in window], dtype=float)
    angle_y = np.array([row.get("angle_y", 0.0) for row in window], dtype=float)
    angle_z = np.array([row.get("angle_z", 0.0) for row in window], dtype=float)

    wx = np.array([row.get("wx", 0.0) for row in window], dtype=float)
    wy = np.array([row.get("wy", 0.0) for row in window], dtype=float)
    wz = np.array([row.get("wz", 0.0) for row in window], dtype=float)

    range_x = float(np.max(angle_x) - np.min(angle_x))
    range_y = float(np.max(angle_y) - np.min(angle_y))
    range_z = float(np.max(angle_z) - np.min(angle_z))

    range_wx = float(np.max(wx) - np.min(wx))
    range_wy = float(np.max(wy) - np.min(wy))
    range_wz = float(np.max(wz) - np.min(wz))

    # -------------------------------------------------
    # Strong ABD rule:
    # ABD has higher angle_y and much higher wz.
    # -------------------------------------------------
    if range_wz >= 285 and range_y >= 42:
        return "ABD"

    if range_wz >= 330:
        return "ABD"

    # -------------------------------------------------
    # Strong FEL rule:
    # FEL has higher angle_x, lower angle_y, and lower wz.
    # -------------------------------------------------
    if range_x >= 120 and range_y <= 43 and range_wz <= 285:
        return "FEL"

    if range_wz <= 250 and range_x > range_y * 2.5:
        return "FEL"

    # -------------------------------------------------
    # Score-based fallback
    # -------------------------------------------------
    abd_score = 0
    fel_score = 0

    # ABD indicators
    if range_y >= 42:
        abd_score += 2

    if range_wz >= 285:
        abd_score += 3

    if range_wz > range_wy:
        abd_score += 1

    # FEL indicators
    if range_x >= 120:
        fel_score += 2

    if range_y <= 43:
        fel_score += 1

    if range_wz <= 285:
        fel_score += 3

    if range_x > range_y * 2.5:
        fel_score += 1

    if abd_score > fel_score:
        return "ABD"

    if fel_score > abd_score:
        return "FEL"

    return prediction


# =====================================================
# PREDICTION + FEEDBACK
# =====================================================
def generate_feedback(prediction, confidence):
    prediction = str(prediction)

    if prediction == "REST":
        return "Rest detected. User is not performing an exercise."

    if prediction == "WRONG":
        return "Incorrect movement detected. Adjust posture and repeat slowly."

    if confidence < 60:
        return "Low confidence. Movement is unclear or sensor placement may be unstable."

    if prediction == "ABD":
        return "Abduction detected. Side arm elevation pattern identified."

    if prediction == "FEL":
        return "Forward elevation detected. Forward arm lifting pattern identified."

    if prediction == "IR":
        return "Internal rotation detected. Keep elbow bent and rotate inward slowly."

    if prediction == "PEN":
        return "Pendulum detected. Keep the arm relaxed with gentle swinging."

    return "Movement detected."


def fallback_prediction(window):
    if len(window) == 0:
        return "Waiting", 0, "No data received."

    angle_x = np.array([row.get("angle_x", 0.0) for row in window])
    angle_y = np.array([row.get("angle_y", 0.0) for row in window])
    angle_z = np.array([row.get("angle_z", 0.0) for row in window])

    range_x = float(np.max(angle_x) - np.min(angle_x))
    range_y = float(np.max(angle_y) - np.min(angle_y))
    range_z = float(np.max(angle_z) - np.min(angle_z))

    ranges = {
        "x": range_x,
        "y": range_y,
        "z": range_z,
    }

    dominant_axis = max(ranges, key=ranges.get)
    dominant_range = ranges[dominant_axis]

    if dominant_range < 5:
        pred = "REST"
        conf = 70
    elif dominant_axis == "x":
        pred = "FEL"
        conf = 75
    elif dominant_axis == "y":
        pred = "ABD"
        conf = 75
    else:
        pred = "IR"
        conf = 70

    return pred, conf, "Fallback movement rule used."


def predict_from_window():
    global latest_prediction

    window = list(live_window)

    if len(window) < WINDOW_SIZE:
        latest_prediction = {
            "exercise": "Collecting",
            "confidence": 0,
            "feedback": f"Collecting sensor samples: {len(window)}/{WINDOW_SIZE}",
        }
        return latest_prediction

    if model is None or scaler is None or feature_names is None:
        pred, conf, fb = fallback_prediction(window)

        pred = correct_abd_fel_prediction(pred, window)

        latest_prediction = {
            "exercise": pred,
            "confidence": conf,
            "feedback": fb,
        }
        return latest_prediction

    try:
        X, missing = make_model_input(window)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            conf = float(np.max(probs) * 100)
        else:
            conf = 85.0

        original_pred = str(pred)

        pred = correct_abd_fel_prediction(pred, window)

        feedback = generate_feedback(pred, conf)

        if original_pred != str(pred):
            feedback += f" Direction correction applied: {original_pred} → {pred}."

        if missing > 0:
            feedback += f" Missing live features filled: {missing}."

        latest_prediction = {
            "exercise": str(pred),
            "confidence": round(conf, 1),
            "feedback": feedback,
        }

        return latest_prediction

    except Exception as e:
        pred, conf, fb = fallback_prediction(window)

        pred = correct_abd_fel_prediction(pred, window)

        latest_prediction = {
            "exercise": pred,
            "confidence": conf,
            "feedback": f"{fb} Model error: {str(e)}",
        }

        return latest_prediction


# =====================================================
# BLE SENSOR LOOP
# =====================================================
async def sensor_loop():
    global latest_reading
    global sensor_running
    global sensor_connected
    global sensor_status

    buffer = bytearray()

    def notification_handler(sender, data):
        global latest_reading

        buffer.extend(data)

        while len(buffer) >= 20:
            if not (buffer[0] == 0x55 and buffer[1] == 0x61):
                buffer.pop(0)
                continue

            packet = buffer[:20]
            del buffer[:20]

            sample = parse_wt901_packet(packet)

            if sample is not None:
                latest_reading = sample
                live_window.append(sample)

    try:
        sensor_status = "Connecting to WT901BLE..."
        print(sensor_status)

        async with BleakClient(ADDRESS) as client:
            sensor_connected = True
            sensor_running = True

            sensor_status = "Connected to WT901BLE"
            print(sensor_status)

            await client.start_notify(NOTIFY_UUID, notification_handler)

            sensor_status = "Receiving live WT901BLE data"
            print(sensor_status)

            while not stop_event.is_set():
                await asyncio.sleep(0.2)

            await client.stop_notify(NOTIFY_UUID)

    except Exception as e:
        sensor_status = f"Sensor error: {str(e)}"
        print(sensor_status)

    finally:
        sensor_connected = False
        sensor_running = False
        sensor_status = "Sensor stopped"


def run_sensor_thread():
    asyncio.run(sensor_loop())


# =====================================================
# ROUTES
# =====================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "Rehab Fit backend is running",
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "features_loaded": feature_names is not None,
            "feature_count": len(feature_names) if feature_names is not None else 0,
            "sensor_status": sensor_status,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "features_loaded": feature_names is not None,
            "sensor_running": sensor_running,
            "sensor_connected": sensor_connected,
            "sensor_status": sensor_status,
        }
    )


@app.route("/start_sensor", methods=["POST"])
def start_sensor():
    global sensor_thread
    global sensor_status

    if sensor_running:
        return jsonify(
            {
                "message": "Sensor already running",
                "sensor_status": sensor_status,
            }
        )

    live_window.clear()
    stop_event.clear()

    sensor_thread = threading.Thread(target=run_sensor_thread, daemon=True)
    sensor_thread.start()

    sensor_status = "Sensor thread started"

    return jsonify(
        {
            "message": "Started sensor",
            "sensor_status": sensor_status,
        }
    )


@app.route("/stop_sensor", methods=["POST"])
def stop_sensor():
    global sensor_status

    stop_event.set()
    sensor_status = "Stopping sensor..."

    return jsonify(
        {
            "message": "Stopping sensor",
            "sensor_status": sensor_status,
        }
    )


@app.route("/latest", methods=["GET"])
def latest():
    return jsonify(
        {
            "reading": latest_reading,
            "samples": len(live_window),
            "sensor_running": sensor_running,
            "sensor_connected": sensor_connected,
            "sensor_status": sensor_status,
        }
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = predict_from_window()
    return jsonify(result)


@app.route("/clear", methods=["POST"])
def clear():
    live_window.clear()
    return jsonify({"message": "Live window cleared"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)