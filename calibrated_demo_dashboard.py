import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os

try:
    import serial
except ImportError:
    serial = None


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Rehab Fit Calibration Demo",
    page_icon="💪",
    layout="wide"
)

st.title("💪 Rehab Fit - Personalized Calibration Demo")
st.caption("WIT IMU Sensor + User-Specific Baseline Feedback")


# -----------------------------
# SETTINGS
# -----------------------------
BASELINE_FILE = "exercise_baselines.json"

EXERCISES = {
    "ABD": "Abduction",
    "FEL": "Forward Elevation",
    "PEN": "Pendulum",
    "IR": "Internal Rotation"
}

ANGLE_COLUMNS = ["angle_x", "angle_y", "angle_z"]


# -----------------------------
# LOAD / SAVE BASELINES
# -----------------------------
def load_baselines():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_baselines(baselines):
    with open(BASELINE_FILE, "w") as f:
        json.dump(baselines, f, indent=4)


# -----------------------------
# SERIAL READING
# -----------------------------
def parse_sensor_line(line):
    """
    Expected format:
    ax,ay,az,wx,wy,wz,angle_x,angle_y,angle_z

    Example:
    0.12,0.45,9.71,1.2,0.5,0.1,12.5,4.3,80.2
    """

    try:
        parts = line.strip().split(",")

        if len(parts) < 9:
            return None

        values = [float(x) for x in parts[:9]]

        return {
            "ax": values[0],
            "ay": values[1],
            "az": values[2],
            "wx": values[3],
            "wy": values[4],
            "wz": values[5],
            "angle_x": values[6],
            "angle_y": values[7],
            "angle_z": values[8],
            "timestamp": time.time()
        }

    except:
        return None


def read_sensor_window(port, baudrate, duration=4, demo_mode=False):
    """
    Reads sensor values for a few seconds.
    If demo_mode=True, it generates fake movement data so dashboard can still be tested.
    """

    data = []
    start_time = time.time()

    if demo_mode:
        while time.time() - start_time < duration:
            t = time.time() - start_time

            # Fake smooth movement
            angle_x = 20 + 60 * abs(np.sin(t))
            angle_y = 10 + 20 * abs(np.sin(t * 0.8))
            angle_z = 5 + 10 * abs(np.sin(t * 0.5))

            data.append({
                "ax": np.random.normal(0, 0.2),
                "ay": np.random.normal(0, 0.2),
                "az": np.random.normal(9.8, 0.2),
                "wx": np.random.normal(0, 0.5),
                "wy": np.random.normal(0, 0.5),
                "wz": np.random.normal(0, 0.5),
                "angle_x": angle_x,
                "angle_y": angle_y,
                "angle_z": angle_z,
                "timestamp": time.time()
            })

            time.sleep(0.05)

        return pd.DataFrame(data)

    if serial is None:
        st.error("pyserial is not installed. Run: pip install pyserial")
        return pd.DataFrame()

    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(1)

        while time.time() - start_time < duration:
            line = ser.readline().decode(errors="ignore").strip()

            if line:
                row = parse_sensor_line(line)
                if row:
                    data.append(row)

        ser.close()

    except Exception as e:
        st.error(f"Could not read sensor: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data)


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_rep_features(df):
    if df.empty:
        return None

    features = {}

    for col in ANGLE_COLUMNS:
        features[f"{col}_min"] = float(df[col].min())
        features[f"{col}_max"] = float(df[col].max())
        features[f"{col}_range"] = float(df[col].max() - df[col].min())
        features[f"{col}_mean"] = float(df[col].mean())

    # Find dominant angle axis
    ranges = {
        "angle_x": features["angle_x_range"],
        "angle_y": features["angle_y_range"],
        "angle_z": features["angle_z_range"]
    }

    dominant_axis = max(ranges, key=ranges.get)
    dominant_range = ranges[dominant_axis]

    features["dominant_axis"] = dominant_axis
    features["dominant_range"] = float(dominant_range)

    return features


def compare_with_baseline(current, baseline):
    """
    Compares current rep with saved baseline.
    """

    if current is None or baseline is None:
        return 0, "No data", "Please calibrate first."

    baseline_range = baseline["dominant_range"]
    current_range = current["dominant_range"]

    # Avoid divide by zero
    if baseline_range < 1:
        baseline_range = 1

    range_difference = abs(current_range - baseline_range)
    similarity = max(0, 100 - ((range_difference / baseline_range) * 100))

    same_axis = current["dominant_axis"] == baseline["dominant_axis"]

    # Penalize if movement is mostly happening in another axis
    if not same_axis:
        similarity -= 25

    similarity = max(0, min(100, similarity))

    if similarity >= 80:
        status = "✅ Correct"
        feedback = "Good movement. Your rep is close to your calibrated baseline."
    elif similarity >= 60:
        status = "⚠️ Needs Improvement"
        feedback = "Movement is similar, but range or direction is not fully matching."
    else:
        status = "❌ Incorrect"
        feedback = "Movement does not match the calibrated correct rep. Try again slowly."

    return round(similarity, 1), status, feedback


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Sensor Settings")

demo_mode = st.sidebar.checkbox("Demo mode without sensor", value=False)

port = st.sidebar.text_input("COM Port", value="COM3")
baudrate = st.sidebar.selectbox("Baudrate", [9600, 115200, 57600, 38400], index=0)

duration = st.sidebar.slider("Recording duration per rep", 2, 8, 4)

selected_code = st.sidebar.selectbox(
    "Select Exercise",
    list(EXERCISES.keys()),
    format_func=lambda x: f"{x} - {EXERCISES[x]}"
)

st.sidebar.info(
    "Use calibration first. Perform one correct rep, then test the next rep."
)


# -----------------------------
# MAIN UI
# -----------------------------
baselines = load_baselines()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1️⃣ Calibration")

    st.write(f"Selected exercise: **{selected_code} - {EXERCISES[selected_code]}**")

    if st.button("Record Correct Trial Rep", use_container_width=True):
        st.warning("Perform ONE correct rep now...")

        progress = st.progress(0)

        for i in range(100):
            time.sleep(duration / 100)
            progress.progress(i + 1)

        df = read_sensor_window(port, baudrate, duration, demo_mode)

        if df.empty:
            st.error("No sensor data recorded.")
        else:
            baseline_features = extract_rep_features(df)

            baselines[selected_code] = baseline_features
            save_baselines(baselines)

            st.success("Baseline saved successfully!")

            st.write("Saved baseline:")
            st.json(baseline_features)

            st.line_chart(df[ANGLE_COLUMNS])

with col2:
    st.subheader("2️⃣ Test Rep")

    if st.button("Record Test Rep", use_container_width=True):
        if selected_code not in baselines:
            st.error("Please record a calibration rep first.")
        else:
            st.warning("Perform the test rep now...")

            progress = st.progress(0)

            for i in range(100):
                time.sleep(duration / 100)
                progress.progress(i + 1)

            df = read_sensor_window(port, baudrate, duration, demo_mode)

            if df.empty:
                st.error("No sensor data recorded.")
            else:
                current_features = extract_rep_features(df)
                baseline_features = baselines[selected_code]

                similarity, status, feedback = compare_with_baseline(
                    current_features,
                    baseline_features
                )

                st.metric("Movement Similarity", f"{similarity}%")
                st.subheader(status)
                st.write(feedback)

                st.write("Current rep features:")
                st.json(current_features)

                st.line_chart(df[ANGLE_COLUMNS])


# -----------------------------
# BASELINE SUMMARY
# -----------------------------
st.divider()
st.subheader("Saved Exercise Baselines")

if baselines:
    summary_rows = []

    for code, features in baselines.items():
        summary_rows.append({
            "Exercise": f"{code} - {EXERCISES.get(code, code)}",
            "Dominant Axis": features.get("dominant_axis", "-"),
            "Dominant Range": round(features.get("dominant_range", 0), 2),
            "Angle X Range": round(features.get("angle_x_range", 0), 2),
            "Angle Y Range": round(features.get("angle_y_range", 0), 2),
            "Angle Z Range": round(features.get("angle_z_range", 0), 2)
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

else:
    st.info("No baselines saved yet.")


# -----------------------------
# EXPLANATION
# -----------------------------
st.divider()
st.subheader("How this works")

st.write(
    """
    This dashboard uses a personalized calibration method.

    First, the user performs one correct trial repetition. The system records the angle range
    and movement direction from the WIT IMU sensor. This becomes the baseline for that exercise.

    During the test repetition, the current movement is compared with the saved baseline.
    If the movement range and direction are similar, the system marks it as correct.
    If the movement is too different, the system gives feedback.
    """
)