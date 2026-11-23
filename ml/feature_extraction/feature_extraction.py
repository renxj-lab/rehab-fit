import pandas as pd
import numpy as np
from pathlib import Path

base_path = Path(r"C:\Users\john\OneDrive\المستندات\thesis")
dataset_path = base_path / "dataset"

train_path = dataset_path / "train_dataset.csv"
test_path = dataset_path / "test_dataset.csv"

sensor_cols = ["ax", "ay", "az", "wx", "wy", "wz"]

# 50 Hz dataset
# 100 samples = 2 seconds
window_size = 50
step_size = 25   # 50% overlap


def rms(series):
    return np.sqrt(np.mean(series ** 2))


def extract_features(df):
    features = []

    for source_file, group in df.groupby("source_file"):
        group = group.reset_index(drop=True)

        for start in range(0, len(group) - window_size + 1, step_size):
            window = group.iloc[start:start + window_size]

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

            row["subject"] = window["subject"].iloc[0]
            row["exercise_code"] = window["exercise_code"].iloc[0]
            row["exercise"] = window["exercise"].iloc[0]
            row["side"] = window["side"].iloc[0]
            row["source_file"] = source_file

            features.append(row)

    feature_df = pd.DataFrame(features)
    feature_df = feature_df.fillna(0)

    return feature_df


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

features_train = extract_features(train_df)
features_test = extract_features(test_df)

features_train.to_csv(dataset_path / "features_train.csv", index=False)
features_test.to_csv(dataset_path / "features_test.csv", index=False)

print("Feature extraction completed.")
print("Train features shape:", features_train.shape)
print("Test features shape:", features_test.shape)
