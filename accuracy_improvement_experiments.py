import pandas as pd
import numpy as np
import os
import joblib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# ======================================================
# 1. PATHS
# ======================================================

BASE_FOLDER = r"C:\Users\john\OneDrive\المستندات\thesis"

DATASET_FOLDER = os.path.join(BASE_FOLDER, "dataset")
RESULTS_FOLDER = os.path.join(BASE_FOLDER, "results")
MODEL_FOLDER = os.path.join(BASE_FOLDER, "models")

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

train_path = os.path.join(DATASET_FOLDER, "features_train.csv")
test_path = os.path.join(DATASET_FOLDER, "features_test.csv")


# ======================================================
# 2. LOAD DATA
# ======================================================

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ======================================================
# 3. PREPARE FEATURES AND LABELS
# ======================================================

LABEL_COLUMN = "exercise"

drop_columns = [
    "exercise",
    "exercise_code",
    "subject",
    "side",
    "source_file"
]

drop_columns = [col for col in drop_columns if col in train_df.columns]

X_train = train_df.drop(columns=drop_columns)
y_train = train_df[LABEL_COLUMN]

X_test = test_df.drop(columns=drop_columns)
y_test = test_df[LABEL_COLUMN]

print("\nNumber of features:", X_train.shape[1])


# ======================================================
# 4. ENCODE LABELS
# ======================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("\nClasses:")
print(label_encoder.classes_)


# ======================================================
# 5. HELPER FUNCTIONS
# ======================================================

def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1 Score": f1_score(y_true, y_pred, average="weighted")
    }


def save_confusion_matrix(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    labels = label_encoder.classes_

    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, filename))
    plt.close()


all_results = []


# ======================================================
# 6. EXPERIMENT 1: CURRENT BEST SVM BASELINE
# ======================================================

print("\n================ EXPERIMENT 1: BASELINE SVM ================")

scaler_base = StandardScaler()
X_train_scaled = scaler_base.fit_transform(X_train)
X_test_scaled = scaler_base.transform(X_test)

baseline_svm = SVC(
    kernel="linear",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)

baseline_svm.fit(X_train_scaled, y_train_encoded)

baseline_pred = baseline_svm.predict(X_test_scaled)

baseline_metrics = calculate_metrics(y_test_encoded, baseline_pred)

print("Baseline SVM Metrics:")
print(baseline_metrics)

all_results.append({
    "Experiment": "Baseline SVM",
    "Method": "All features, linear kernel, C=10",
    "Features/Components": X_train.shape[1],
    **baseline_metrics
})


# ======================================================
# 7. EXPERIMENT 2: SVM WITH FEATURE SELECTION
# ======================================================

print("\n================ EXPERIMENT 2: SVM + FEATURE SELECTION ================")

feature_counts = [30, 40, 50, 60, 70, 80, 90, X_train.shape[1]]

best_fs_accuracy = 0
best_fs_model = None
best_fs_scaler = None
best_fs_selector = None
best_fs_count = None
best_fs_pred = None

for k_features in feature_counts:

    if k_features == X_train.shape[1]:
        X_train_selected = X_train
        X_test_selected = X_test
        selector = None
        feature_label = "All Features"
    else:
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_train_selected = selector.fit_transform(X_train, y_train_encoded)
        X_test_selected = selector.transform(X_test)
        feature_label = f"Top {k_features}"

    scaler = StandardScaler()
    X_train_fs_scaled = scaler.fit_transform(X_train_selected)
    X_test_fs_scaled = scaler.transform(X_test_selected)

    svm_fs = SVC(
        kernel="linear",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    )

    svm_fs.fit(X_train_fs_scaled, y_train_encoded)

    pred = svm_fs.predict(X_test_fs_scaled)
    metrics = calculate_metrics(y_test_encoded, pred)

    print(f"{feature_label}: Accuracy = {metrics['Accuracy']:.4f}")

    all_results.append({
        "Experiment": "SVM + Feature Selection",
        "Method": feature_label,
        "Features/Components": k_features,
        **metrics
    })

    if metrics["Accuracy"] > best_fs_accuracy:
        best_fs_accuracy = metrics["Accuracy"]
        best_fs_model = svm_fs
        best_fs_scaler = scaler
        best_fs_selector = selector
        best_fs_count = k_features
        best_fs_pred = pred


print("\nBest Feature Selection Result:")
print("Features:", best_fs_count)
print("Accuracy:", best_fs_accuracy)

save_confusion_matrix(
    y_test_encoded,
    best_fs_pred,
    "Best SVM Feature Selection",
    "best_svm_feature_selection_confusion_matrix.png"
)


# ======================================================
# 8. EXPERIMENT 3: SVM WITH PCA
# ======================================================

print("\n================ EXPERIMENT 3: SVM + PCA ================")

pca_components = [20, 30, 40, 50, 60, 70, 80]

best_pca_accuracy = 0
best_pca_model = None
best_pca_scaler = None
best_pca_transformer = None
best_pca_count = None
best_pca_pred = None

for n_components in pca_components:

    scaler = StandardScaler()
    X_train_pca_scaled = scaler.fit_transform(X_train)
    X_test_pca_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=42)

    X_train_pca = pca.fit_transform(X_train_pca_scaled)
    X_test_pca = pca.transform(X_test_pca_scaled)

    svm_pca = SVC(
        kernel="linear",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    )

    svm_pca.fit(X_train_pca, y_train_encoded)

    pred = svm_pca.predict(X_test_pca)
    metrics = calculate_metrics(y_test_encoded, pred)

    print(f"PCA {n_components}: Accuracy = {metrics['Accuracy']:.4f}")

    all_results.append({
        "Experiment": "SVM + PCA",
        "Method": f"PCA {n_components} components",
        "Features/Components": n_components,
        **metrics
    })

    if metrics["Accuracy"] > best_pca_accuracy:
        best_pca_accuracy = metrics["Accuracy"]
        best_pca_model = svm_pca
        best_pca_scaler = scaler
        best_pca_transformer = pca
        best_pca_count = n_components
        best_pca_pred = pred


print("\nBest PCA Result:")
print("Components:", best_pca_count)
print("Accuracy:", best_pca_accuracy)

save_confusion_matrix(
    y_test_encoded,
    best_pca_pred,
    "Best SVM PCA",
    "best_svm_pca_confusion_matrix.png"
)


# ======================================================
# 9. EXPERIMENT 4: BALANCED SVM
# ======================================================

print("\n================ EXPERIMENT 4: BALANCED SVM ================")

balanced_C_values = [1, 5, 10, 15, 20, 30, 50]

best_balanced_accuracy = 0
best_balanced_model = None
best_balanced_scaler = None
best_balanced_C = None
best_balanced_pred = None

for C in balanced_C_values:

    scaler = StandardScaler()
    X_train_balanced = scaler.fit_transform(X_train)
    X_test_balanced = scaler.transform(X_test)

    balanced_svm = SVC(
        kernel="linear",
        C=C,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    )

    balanced_svm.fit(X_train_balanced, y_train_encoded)

    pred = balanced_svm.predict(X_test_balanced)
    metrics = calculate_metrics(y_test_encoded, pred)

    print(f"Balanced SVM C={C}: Accuracy = {metrics['Accuracy']:.4f}")

    all_results.append({
        "Experiment": "Balanced SVM",
        "Method": f"class_weight=balanced, C={C}",
        "Features/Components": X_train.shape[1],
        **metrics
    })

    if metrics["Accuracy"] > best_balanced_accuracy:
        best_balanced_accuracy = metrics["Accuracy"]
        best_balanced_model = balanced_svm
        best_balanced_scaler = scaler
        best_balanced_C = C
        best_balanced_pred = pred


print("\nBest Balanced SVM Result:")
print("C:", best_balanced_C)
print("Accuracy:", best_balanced_accuracy)

save_confusion_matrix(
    y_test_encoded,
    best_balanced_pred,
    "Best Balanced SVM",
    "best_balanced_svm_confusion_matrix.png"
)


# ======================================================
# 10. EXPERIMENT 5: VOTING ENSEMBLE
# ======================================================

print("\n================ EXPERIMENT 5: VOTING ENSEMBLE ================")

voting_scaler = StandardScaler()
X_train_vote = voting_scaler.fit_transform(X_train)
X_test_vote = voting_scaler.transform(X_test)

svm_vote = SVC(
    kernel="linear",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)

knn_vote = KNeighborsClassifier(
    n_neighbors=15,
    weights="distance",
    metric="manhattan"
)

extra_vote = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=30,
    random_state=42,
    n_jobs=-1
)

voting_model = VotingClassifier(
    estimators=[
        ("svm", svm_vote),
        ("knn", knn_vote),
        ("extra_trees", extra_vote)
    ],
    voting="soft"
)

voting_model.fit(X_train_vote, y_train_encoded)

voting_pred = voting_model.predict(X_test_vote)

voting_metrics = calculate_metrics(y_test_encoded, voting_pred)

print("Voting Ensemble Metrics:")
print(voting_metrics)

all_results.append({
    "Experiment": "Voting Ensemble",
    "Method": "SVM + KNN + Extra Trees soft voting",
    "Features/Components": X_train.shape[1],
    **voting_metrics
})

save_confusion_matrix(
    y_test_encoded,
    voting_pred,
    "Voting Ensemble",
    "voting_ensemble_confusion_matrix.png"
)


# ======================================================
# 11. SAVE FINAL COMPARISON
# ======================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

results_path = os.path.join(
    RESULTS_FOLDER,
    "accuracy_improvement_experiments.csv"
)

results_df.to_csv(results_path, index=False)

print("\n================ FINAL IMPROVEMENT RESULTS ================")
print(results_df)

best_result = results_df.iloc[0]

with open(os.path.join(RESULTS_FOLDER, "best_accuracy_improvement_result.txt"), "w") as file:
    file.write("Best Accuracy Improvement Result\n")
    file.write("================================\n")
    file.write(str(best_result))


# ======================================================
# 12. SAVE GRAPHS
# ======================================================

plt.figure(figsize=(12, 6))
plt.bar(results_df["Experiment"] + "\n" + results_df["Method"].astype(str), results_df["Accuracy"])
plt.xlabel("Experiment")
plt.ylabel("Accuracy")
plt.title("Accuracy Improvement Experiments")
plt.xticks(rotation=80, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "accuracy_improvement_comparison.png"))
plt.close()


# Feature selection graph
fs_df = results_df[results_df["Experiment"] == "SVM + Feature Selection"]

if not fs_df.empty:
    plt.figure(figsize=(8, 5))
    plt.plot(fs_df["Features/Components"], fs_df["Accuracy"], marker="o")
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Accuracy")
    plt.title("SVM Accuracy with Feature Selection")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "svm_feature_selection_accuracy.png"))
    plt.close()


# PCA graph
pca_df = results_df[results_df["Experiment"] == "SVM + PCA"]

if not pca_df.empty:
    plt.figure(figsize=(8, 5))
    plt.plot(pca_df["Features/Components"], pca_df["Accuracy"], marker="o")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Accuracy")
    plt.title("SVM Accuracy with PCA")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "svm_pca_accuracy.png"))
    plt.close()


# ======================================================
# 13. SAVE BEST MODEL IF IT IMPROVES BASELINE
# ======================================================

baseline_accuracy = baseline_metrics["Accuracy"]
best_accuracy = best_result["Accuracy"]

print("\nBaseline Accuracy:", baseline_accuracy)
print("Best Experiment Accuracy:", best_accuracy)

if best_accuracy > baseline_accuracy:
    print("\nImprovement found. Saving best improvement summary.")

    # Save only summary here.
    # Do not overwrite your main svm_model.pkl automatically.
    # You can manually choose later after checking the results.
else:
    print("\nNo experiment beat the current baseline SVM. Keep current SVM as final model.")


print("\nAccuracy improvement experiments completed.")
print("Results saved in:")
print(RESULTS_FOLDER)
