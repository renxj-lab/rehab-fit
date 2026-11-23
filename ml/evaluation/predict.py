import pandas as pd
import numpy as np
import joblib

sensor_cols = ["ax", "ay", "az", "wx", "wy", "wz"]
window_size = 100
step_size = 50

files = {
    "abd2.csv": "ABD",
    "forwardel2.csv": "FEL",
    "ir2.csv": "IR",
    "pedulum2.csv": "PEN",
}

def rms(series):
    return np.sqrt(np.mean(series ** 2))

def convert_wt901(df):
    df = df.rename(columns={
        "Acceleration X(g)": "ax",
        "Acceleration Y(g)": "ay",
        "Acceleration Z(g)": "az",
        "Angular velocity X(°/s)": "wx",
        "Angular velocity Y(°/s)": "wy",
        "Angular velocity Z(°/s)": "wz",
    })

    return df[sensor_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

def extract_features_from_window(window):
    row = {}

    for col in sensor_cols:
        row[f"{col}_mean"] = window[col].mean()
        row[f"{col}_std"] = window[col].std()
        row[f"{col}_min"] = window[col].min()
        row[f"{col}_max"] = window[col].max()
        row[f"{col}_rms"] = rms(window[col])
        row[f"{col}_var"] = window[col].var()
        row[f"{col}_median"] = window[col].median()
        row[f"{col}_range"] = window[col].max() - window[col].min()
        row[f"{col}_energy"] = np.sum(window[col] ** 2)
        row[f"{col}_skew"] = window[col].skew()
        row[f"{col}_kurtosis"] = window[col].kurtosis()
        row[f"{col}_iqr"] = window[col].quantile(0.75) - window[col].quantile(0.25)
        row[f"{col}_zero_crossings"] = ((window[col].shift(1) * window[col]) < 0).sum()
        row[f"{col}_mav"] = np.mean(np.abs(window[col]))

    acc_mag = np.sqrt(window["ax"]**2 + window["ay"]**2 + window["az"]**2)
    gyro_mag = np.sqrt(window["wx"]**2 + window["wy"]**2 + window["wz"]**2)

    row["acc_mag_mean"] = acc_mag.mean()
    row["acc_mag_std"] = acc_mag.std()
    row["acc_mag_min"] = acc_mag.min()
    row["acc_mag_max"] = acc_mag.max()
    row["acc_mag_rms"] = rms(acc_mag)
    row["acc_mag_energy"] = np.sum(acc_mag ** 2)

    row["gyro_mag_mean"] = gyro_mag.mean()
    row["gyro_mag_std"] = gyro_mag.std()
    row["gyro_mag_min"] = gyro_mag.min()
    row["gyro_mag_max"] = gyro_mag.max()
    row["gyro_mag_rms"] = rms(gyro_mag)
    row["gyro_mag_energy"] = np.sum(gyro_mag ** 2)

    row["sma_acc"] = (
        np.sum(np.abs(window["ax"])) +
        np.sum(np.abs(window["ay"])) +
        np.sum(np.abs(window["az"]))
    ) / len(window)

    row["sma_gyro"] = (
        np.sum(np.abs(window["wx"])) +
        np.sum(np.abs(window["wy"])) +
        np.sum(np.abs(window["wz"]))
    ) / len(window)

    row["corr_ax_ay"] = window["ax"].corr(window["ay"])
    row["corr_ax_az"] = window["ax"].corr(window["az"])
    row["corr_ay_az"] = window["ay"].corr(window["az"])

    row["corr_wx_wy"] = window["wx"].corr(window["wy"])
    row["corr_wx_wz"] = window["wx"].corr(window["wz"])
    row["corr_wy_wz"] = window["wy"].corr(window["wz"])

    return row

model = joblib.load("svm_4class_model.pkl")
scaler = joblib.load("svm_4class_scaler.pkl")

for file_name, true_label in files.items():
    print("\n------------------------------")
    print(f"Reading: {file_name}")

    df_raw = pd.read_csv(file_name)
    df = convert_wt901(df_raw)

    feature_rows = []

    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        feature_rows.append(extract_features_from_window(window))

    features = pd.DataFrame(feature_rows).fillna(0)

    expected_features = scaler.feature_names_in_
    features = features.reindex(columns=expected_features, fill_value=0)

    X_scaled = scaler.transform(features)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    final_pred = pd.Series(preds).mode()[0]
    confidence = probs.max(axis=1).mean() * 100

    print(f"Actual: {true_label}")
    print(f"Predicted: {final_pred}")
    print(f"Confidence: {confidence:.2f}%")
