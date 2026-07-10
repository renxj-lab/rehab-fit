import streamlit as st
import pandas as pd
import numpy as np
import joblib
import asyncio
import struct
import time
import os
from bleak import BleakClient


# =====================================================
# SETTINGS
# =====================================================
ADDRESS = "E1:05:3D:86:3A:03"
NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "wt901_live_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "wt901_live_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "wt901_live_features.pkl")


# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="Live Rehab FitBand",
    page_icon="💪",
    layout="wide"
)


# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)


# =====================================================
# STYLE
# =====================================================
st.markdown("""
<style>
.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    color: #00D1FF;
}
.sub-title {
    font-size: 18px;
    text-align: center;
    color: #CCCCCC;
    margin-bottom: 25px;
}
.correct-box {
    padding: 30px;
    border-radius: 18px;
    background-color: #123D2A;
    color: #8CFFB0;
    font-size: 30px;
    font-weight: bold;
    text-align: center;
}
.wrong-box {
    padding: 30px;
    border-radius: 18px;
    background-color: #4A1F1F;
    color: #FF9A9A;
    font-size: 30px;
    font-weight: bold;
    text-align: center;
}
.warning-box {
    padding: 30px;
    border-radius: 18px;
    background-color: #4A3A1F;
    color: #FFD27A;
    font-size: 30px;
    font-weight: bold;
    text-align: center;
}
.small-card {
    padding: 16px;
    border-radius: 12px;
    background-color: #F5F7FA;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">Live Smart Physiotherapy FitBand</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">WT901BLE Real-Time Exercise Classification, Correctness Feedback, and Rep Counting</div>',
    unsafe_allow_html=True
)


# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features(window):
    features = {}

    for col in window.columns:
        values = window[col].values

        features[f"{col}_mean"] = np.mean(values)
        features[f"{col}_std"] = np.std(values)
        features[f"{col}_min"] = np.min(values)
        features[f"{col}_max"] = np.max(values)
        features[f"{col}_rms"] = np.sqrt(np.mean(values ** 2))
        features[f"{col}_var"] = np.var(values)
        features[f"{col}_median"] = np.median(values)
        features[f"{col}_range"] = np.max(values) - np.min(values)
        features[f"{col}_energy"] = np.sum(values ** 2)

        try:
            features[f"{col}_skew"] = pd.Series(values).skew()
        except:
            features[f"{col}_skew"] = 0

        try:
            features[f"{col}_kurtosis"] = pd.Series(values).kurtosis()
        except:
            features[f"{col}_kurtosis"] = 0

    if all(col in window.columns for col in ["ax", "ay", "az"]):
        acc_mag = np.sqrt(
            window["ax"].values ** 2 +
            window["ay"].values ** 2 +
            window["az"].values ** 2
        )

        features["acc_mag_mean"] = np.mean(acc_mag)
        features["acc_mag_std"] = np.std(acc_mag)
        features["acc_mag_min"] = np.min(acc_mag)
        features["acc_mag_max"] = np.max(acc_mag)
        features["acc_mag_range"] = np.max(acc_mag) - np.min(acc_mag)
        features["acc_mag_energy"] = np.sum(acc_mag ** 2)

    if all(col in window.columns for col in ["wx", "wy", "wz"]):
        gyro_mag = np.sqrt(
            window["wx"].values ** 2 +
            window["wy"].values ** 2 +
            window["wz"].values ** 2
        )

        features["gyro_mag_mean"] = np.mean(gyro_mag)
        features["gyro_mag_std"] = np.std(gyro_mag)
        features["gyro_mag_min"] = np.min(gyro_mag)
        features["gyro_mag_max"] = np.max(gyro_mag)
        features["gyro_mag_range"] = np.max(gyro_mag) - np.min(gyro_mag)
        features["gyro_mag_energy"] = np.sum(gyro_mag ** 2)

    if all(col in window.columns for col in ["angle_x", "angle_y", "angle_z"]):
        angle_mag = np.sqrt(
            window["angle_x"].values ** 2 +
            window["angle_y"].values ** 2 +
            window["angle_z"].values ** 2
        )

        features["angle_mag_mean"] = np.mean(angle_mag)
        features["angle_mag_std"] = np.std(angle_mag)
        features["angle_mag_min"] = np.min(angle_mag)
        features["angle_mag_max"] = np.max(angle_mag)
        features["angle_mag_range"] = np.max(angle_mag) - np.min(angle_mag)
        features["angle_mag_energy"] = np.sum(angle_mag ** 2)

    feature_df = pd.DataFrame([features])

    for feature in feature_names:
        if feature not in feature_df.columns:
            feature_df[feature] = 0

    feature_df = feature_df[feature_names]

    feature_df = feature_df.replace([np.inf, -np.inf], 0)
    feature_df = feature_df.fillna(0)

    return feature_df


# =====================================================
# REP COUNTING
# =====================================================
def count_reps(angle_df, selected_exercise):
    if angle_df is None or len(angle_df) < 20:
        return 0, "Not enough data"

    ranges = {
        "angle_x": angle_df["angle_x"].max() - angle_df["angle_x"].min(),
        "angle_y": angle_df["angle_y"].max() - angle_df["angle_y"].min(),
        "angle_z": angle_df["angle_z"].max() - angle_df["angle_z"].min()
    }

    main_axis = max(ranges, key=ranges.get)

    signal = angle_df[main_axis].values
    signal = pd.Series(signal).rolling(window=5, min_periods=1).mean().values

    min_val = np.min(signal)
    max_val = np.max(signal)
    movement_range = max_val - min_val

    if selected_exercise in ["ABD", "FEL"]:
        min_required_range = 10
    elif selected_exercise == "IR":
        min_required_range = 5
    else:
        min_required_range = 4

    if movement_range < min_required_range:
        return 0, main_axis

    low_threshold = min_val + 0.30 * movement_range
    high_threshold = min_val + 0.70 * movement_range

    reps = 0
    state = "low"

    for value in signal:
        if state == "low" and value > high_threshold:
            state = "high"

        elif state == "high" and value < low_threshold:
            reps += 1
            state = "low"

    return reps, main_axis


# =====================================================
# FEEDBACK LOGIC
# =====================================================
def get_feedback(selected_exercise, predicted_exercise, confidence, confidence_limit, angle_df):
    angle_range_x = angle_df["angle_x"].max() - angle_df["angle_x"].min()
    angle_range_y = angle_df["angle_y"].max() - angle_df["angle_y"].min()
    angle_range_z = angle_df["angle_z"].max() - angle_df["angle_z"].min()

    max_angle_range = max(angle_range_x, angle_range_y, angle_range_z)

    if confidence < 45:
        return (
            "UNCERTAIN",
            "The model confidence is very low. Repeat the movement slowly and keep the sensor fixed on your arm."
        )

    if selected_exercise == predicted_exercise:
        if selected_exercise in ["ABD", "FEL"]:
            if max_angle_range >= 10:
                return (
                    "CORRECT",
                    "Correct movement detected. The exercise class and arm movement range are acceptable."
                )
            else:
                return (
                    "INCORRECT",
                    "The exercise was detected, but the arm movement range is too small. Lift your arm more clearly."
                )

        if selected_exercise == "IR":
            if max_angle_range >= 5:
                return (
                    "CORRECT",
                    "Correct internal rotation detected. The rotation range is acceptable."
                )
            else:
                return (
                    "INCORRECT",
                    "Internal rotation was detected, but the rotation range is too small."
                )

        if selected_exercise == "PEN":
            return (
                "CORRECT",
                "Correct pendulum movement detected. Keep the shoulder relaxed and swing gently."
            )

    if selected_exercise in ["ABD", "FEL"] and predicted_exercise in ["ABD", "FEL"]:
        if max_angle_range >= 10 and confidence < 80:
            return (
                "CORRECT",
                "The model confused the lifting direction, but the live movement range is acceptable. Movement is considered correct."
            )
        else:
            return (
                "INCORRECT",
                "The arm lifting movement was detected, but the pattern is not clear enough. Repeat with a smoother motion."
            )

    correction = {
        "ABD": "You selected abduction. Lift your arm sideways in a controlled motion.",
        "FEL": "You selected forward elevation. Lift your arm forward smoothly.",
        "IR": "You selected internal rotation. Rotate the shoulder inward slowly.",
        "PEN": "You selected pendulum. Relax the shoulder and allow gentle swinging."
    }

    return (
        "INCORRECT",
        correction.get(selected_exercise, "The detected movement does not match the selected exercise.")
    )


# =====================================================
# WT901BLE COLLECTOR
# =====================================================
class WT901BLECollector:
    def __init__(self, target_samples):
        self.target_samples = target_samples
        self.buffer = bytearray()
        self.samples = []

    def parse_packet(self, packet):
        if len(packet) != 20:
            return None

        if packet[0] != 0x55 or packet[1] != 0x61:
            return None

        try:
            values = struct.unpack("<hhhhhhhhh", packet[2:20])
        except:
            return None

        ax_raw, ay_raw, az_raw, wx_raw, wy_raw, wz_raw, roll_raw, pitch_raw, yaw_raw = values

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
            "ax": ax,
            "ay": ay,
            "az": az,
            "wx": wx,
            "wy": wy,
            "wz": wz,
            "angle_x": angle_x,
            "angle_y": angle_y,
            "angle_z": angle_z
        }

    def notification_handler(self, sender, data):
        self.buffer.extend(data)

        while len(self.buffer) >= 20:
            if not (self.buffer[0] == 0x55 and self.buffer[1] == 0x61):
                self.buffer.pop(0)
                continue

            packet = self.buffer[:20]
            del self.buffer[:20]

            parsed = self.parse_packet(packet)

            if parsed is not None:
                self.samples.append([
                    parsed["ax"],
                    parsed["ay"],
                    parsed["az"],
                    parsed["wx"],
                    parsed["wy"],
                    parsed["wz"],
                    parsed["angle_x"],
                    parsed["angle_y"],
                    parsed["angle_z"]
                ])


# =====================================================
# BLE DATA COLLECTION
# =====================================================
async def collect_ble_samples(samples):
    collector = WT901BLECollector(samples)

    try:
        async with BleakClient(ADDRESS) as client:
            await client.start_notify(NOTIFY_UUID, collector.notification_handler)

            start_time = time.time()

            while len(collector.samples) < samples:
                await asyncio.sleep(0.02)

                if time.time() - start_time > 25:
                    return None, None, "Timeout: not enough BLE samples received."

            try:
                await client.stop_notify(NOTIFY_UUID)
            except:
                pass

    except Exception as e:
        return None, None, str(e)

    full_df = pd.DataFrame(
        collector.samples[:samples],
        columns=[
            "ax", "ay", "az",
            "wx", "wy", "wz",
            "angle_x", "angle_y", "angle_z"
        ]
    )

    angle_df = full_df[["angle_x", "angle_y", "angle_z"]].copy()

    return full_df, angle_df, None


# =====================================================
# APP LAYOUT
# =====================================================
left, right = st.columns([1, 2])

with left:
    st.subheader("Control Panel")

    selected_exercise = st.selectbox(
        "Select the exercise you are performing",
        ["ABD", "FEL", "IR", "PEN"]
    )

    samples = st.slider(
        "Samples per prediction",
        min_value=100,
        max_value=500,
        value=300,
        step=50
    )

    confidence_limit = st.slider(
        "Minimum confidence",
        min_value=40,
        max_value=95,
        value=60,
        step=5
    )

    start_button = st.button("Start Live Check")

    st.info("""
    ABD = Shoulder Abduction  
    FEL = Forward Elevation  
    IR = Internal Rotation  
    PEN = Pendulum
    """)

    st.write("Sensor: WT901BLE68")
    st.write("Address:", ADDRESS)
    st.write("Model features:", len(feature_names))

with right:
    st.subheader("Live Result")

    if start_button:
        with st.spinner("Collecting live BLE data. Perform the movement now..."):
            live_data, angle_data, error = asyncio.run(collect_ble_samples(samples))

        if error is not None:
            st.error(error)

        else:
            features = extract_features(live_data)
            scaled_features = scaler.transform(features)

            prediction = model.predict(scaled_features)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(scaled_features)[0]
                confidence = np.max(probabilities) * 100
            else:
                confidence = 0

            reps, rep_axis = count_reps(angle_data, selected_exercise)

            status, feedback = get_feedback(
                selected_exercise,
                prediction,
                confidence,
                confidence_limit,
                angle_data
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Selected Exercise", selected_exercise)

            with c2:
                st.metric("Detected Exercise", prediction)

            with c3:
                st.metric("Confidence", f"{confidence:.2f}%")

            with c4:
                st.metric("Reps Counted", reps)

            st.caption(f"Rep counting based on: {rep_axis}")

            st.write("")

            if status == "CORRECT":
                st.markdown(
                    f'<div class="correct-box">CORRECT MOVEMENT ✅<br>{feedback}</div>',
                    unsafe_allow_html=True
                )

            elif status == "INCORRECT":
                st.markdown(
                    f'<div class="wrong-box">INCORRECT MOVEMENT ❌<br>{feedback}</div>',
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f'<div class="warning-box">UNCERTAIN MOVEMENT ⚠️<br>{feedback}</div>',
                    unsafe_allow_html=True
                )

            st.write("")

            st.subheader("Live Sensor Signal")
            st.line_chart(live_data[["ax", "ay", "az", "wx", "wy", "wz"]])

            st.subheader("Live Angle Signal")
            st.line_chart(angle_data)

            st.subheader("Last 20 Sensor Readings")
            st.dataframe(live_data.tail(20), use_container_width=True)

            st.subheader("Angle Range Summary")
            angle_range = angle_data.max() - angle_data.min()
            st.write(angle_range)