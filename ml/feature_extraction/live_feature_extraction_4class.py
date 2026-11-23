import pandas as pd
import numpy as np

sensor_cols = ["ax", "ay", "az", "wx", "wy", "wz"]

def rms(series):
    return np.sqrt(np.mean(series ** 2))

def extract_live_features(window):
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

    feature_df = pd.DataFrame([row])
    feature_df = feature_df.fillna(0)

    return feature_df
