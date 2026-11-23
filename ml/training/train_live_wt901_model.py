import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================================
# SETTINGS
# =====================================================
DATA_FOLDER = "live_training_data"
WINDOW_SIZE = 100
STEP_SIZE = 50

SENSOR_COLS = [
    "ax", "ay", "az",
    "wx", "wy", "wz",
    "angle_x", "angle_y", "angle_z"
]

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features_from_window(window):
    features = {}

    for col in SENSOR_COLS:
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

    # Resultant acceleration magnitude
    acc_mag = np.sqrt(
        window["ax"].values ** 2 +
        window["ay"].values ** 2 +
        window["az"].values ** 2
    )

    gyro_mag = np.sqrt(
        window["wx"].values ** 2 +
        window["wy"].values ** 2 +
        window["wz"].values ** 2
    )

    angle_mag = np.sqrt(
        window["angle_x"].values ** 2 +
        window["angle_y"].values ** 2 +
        window["angle_z"].values ** 2
    )

    for name, values in {
        "acc_mag": acc_mag,
        "gyro_mag": gyro_mag,
        "angle_mag": angle_mag
    }.items():
        features[f"{name}_mean"] = np.mean(values)
        features[f"{name}_std"] = np.std(values)
        features[f"{name}_min"] = np.min(values)
        features[f"{name}_max"] = np.max(values)
        features[f"{name}_range"] = np.max(values) - np.min(values)
        features[f"{name}_energy"] = np.sum(values ** 2)

    return features

# =====================================================
# LOAD LIVE DATA
# =====================================================
def load_live_dataset():
    all_features = []
    all_labels = []

    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in live_training_data folder.")

    for file in csv_files:
        print("Processing:", file)

        df = pd.read_csv(file)

        required_cols = SENSOR_COLS + ["label"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in file {file}")

        df = df.dropna()

        label = df["label"].iloc[0]

        for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
            window = df.iloc[start:start + WINDOW_SIZE]

            features = extract_features_from_window(window)

            all_features.append(features)
            all_labels.append(label)

    X = pd.DataFrame(all_features)
    y = pd.Series(all_labels)

    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X, y

# =====================================================
# TRAIN MODELS
# =====================================================
print("Loading live WT901BLE dataset...")
X, y = load_live_dataset()

print("")
print("Feature dataset shape:", X.shape)
print("Class distribution:")
print(y.value_counts())

feature_names = list(X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

models = {
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42)
}

param_grids = {
    "SVM": {
        "C": [0.1, 1, 5, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "Extra Trees": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
}

results = []
best_model = None
best_model_name = None
best_accuracy = 0

for model_name, model in models.items():
    print("")
    print("Training:", model_name)

    grid = GridSearchCV(
        model,
        param_grids[model_name],
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Best parameters:", grid.best_params_)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    results.append({
        "Model": model_name,
        "Best Parameters": grid.best_params_,
        "Accuracy": accuracy
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = grid.best_estimator_
        best_model_name = model_name

results_df = pd.DataFrame(results)
print("")
print("========== MODEL COMPARISON ==========")
print(results_df)

print("")
print("Best model:", best_model_name)
print("Best accuracy:", best_accuracy)

# =====================================================
# SAVE BEST MODEL
# =====================================================
joblib.dump(best_model, "wt901_live_model.pkl")
joblib.dump(scaler, "wt901_live_scaler.pkl")
joblib.dump(feature_names, "wt901_live_features.pkl")

print("")
print("Saved:")
print("wt901_live_model.pkl")
print("wt901_live_scaler.pkl")
print("wt901_live_features.pkl")
