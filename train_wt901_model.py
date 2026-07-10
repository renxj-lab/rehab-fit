import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sensor_cols = ["ax", "ay", "az", "wx", "wy", "wz"]
window_size = 100
step_size = 25

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

def extract_features(window):
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

all_features = []

for file_name, label in files.items():
    print(f"Processing {file_name} as {label}")

    raw_df = pd.read_csv(file_name)
    df = convert_wt901(raw_df)

    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        row = extract_features(window)
        row["label"] = label
        all_features.append(row)

features_df = pd.DataFrame(all_features).fillna(0)

print("Feature dataset shape:", features_df.shape)
print(features_df["label"].value_counts())

X = features_df.drop(columns=["label"])
y = features_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel="linear", C=5, probability=True, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "wt901_svm_model.pkl")
joblib.dump(scaler, "wt901_svm_scaler.pkl")
joblib.dump(list(X.columns), "wt901_feature_columns.pkl")

print("\nSaved:")
print("wt901_svm_model.pkl")
print("wt901_svm_scaler.pkl")
print("wt901_feature_columns.pkl")