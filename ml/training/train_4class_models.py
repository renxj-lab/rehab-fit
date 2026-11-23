import pandas as pd
import numpy as np
import os
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


BASE_FOLDER = r"C:\Users\john\OneDrive\المستندات\thesis"

DATASET_FOLDER = os.path.join(BASE_FOLDER, "dataset")
MODEL_FOLDER = os.path.join(BASE_FOLDER, "models_4class")
RESULTS_FOLDER = os.path.join(BASE_FOLDER, "results_4class")

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

train_path = os.path.join(DATASET_FOLDER, "features_train_4class.csv")
test_path = os.path.join(DATASET_FOLDER, "features_test_4class.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain labels:")
print(train_df["exercise"].value_counts())

print("\nTest labels:")
print(test_df["exercise"].value_counts())

drop_columns = ["exercise", "exercise_code", "subject", "side", "source_file"]
drop_columns = [col for col in drop_columns if col in train_df.columns]

X_train = train_df.drop(columns=drop_columns)
y_train = train_df["exercise"]

X_test = test_df.drop(columns=drop_columns)
y_test = test_df["exercise"]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

joblib.dump(label_encoder, os.path.join(MODEL_FOLDER, "label_encoder_4class.pkl"))

print("\nClasses:")
print(label_encoder.classes_)


def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1 Score": f1_score(y_true, y_pred, average="weighted")
    }


def save_confusion_matrix(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    plt.imshow(cm)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    labels = label_encoder.classes_
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, filename))
    plt.close()


all_results = []


# =========================
# KNN
# =========================

print("\n========== KNN ==========")

knn_scaler = StandardScaler()
X_train_knn = knn_scaler.fit_transform(X_train)
X_test_knn = knn_scaler.transform(X_test)

best_knn_acc = 0
best_knn = None
best_knn_params = None

for k in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
    for weight in ["uniform", "distance"]:
        for metric in ["euclidean", "manhattan"]:

            model = KNeighborsClassifier(
                n_neighbors=k,
                weights=weight,
                metric=metric
            )

            model.fit(X_train_knn, y_train_encoded)
            pred = model.predict(X_test_knn)
            acc = accuracy_score(y_test_encoded, pred)

            print(f"K={k}, weight={weight}, metric={metric}, acc={acc:.4f}")

            if acc > best_knn_acc:
                best_knn_acc = acc
                best_knn = model
                best_knn_params = (k, weight, metric)

knn_pred = best_knn.predict(X_test_knn)
knn_metrics = get_metrics(y_test_encoded, knn_pred)

save_confusion_matrix(y_test_encoded, knn_pred, "KNN 4-Class", "knn_4class_confusion_matrix.png")

joblib.dump(best_knn, os.path.join(MODEL_FOLDER, "knn_4class_model.pkl"))
joblib.dump(knn_scaler, os.path.join(MODEL_FOLDER, "knn_4class_scaler.pkl"))

all_results.append({
    "Model": "KNN",
    "Best Parameters": str(best_knn_params),
    **knn_metrics
})


# =========================
# RANDOM FOREST
# =========================

print("\n========== RANDOM FOREST ==========")

best_rf_acc = 0
best_rf = None
best_rf_params = None

for n in [100, 200, 300, 500]:
    for depth in [None, 20, 30, 40]:
        for leaf in [1, 2]:
            for max_features in ["sqrt", "log2", None]:

                model = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train_encoded)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test_encoded, pred)

                print(f"Trees={n}, depth={depth}, leaf={leaf}, max_features={max_features}, acc={acc:.4f}")

                if acc > best_rf_acc:
                    best_rf_acc = acc
                    best_rf = model
                    best_rf_params = (n, depth, leaf, max_features)

rf_pred = best_rf.predict(X_test)
rf_metrics = get_metrics(y_test_encoded, rf_pred)

save_confusion_matrix(y_test_encoded, rf_pred, "Random Forest 4-Class", "random_forest_4class_confusion_matrix.png")

joblib.dump(best_rf, os.path.join(MODEL_FOLDER, "random_forest_4class_model.pkl"))

all_results.append({
    "Model": "Random Forest",
    "Best Parameters": str(best_rf_params),
    **rf_metrics
})


# =========================
# SVM
# =========================

print("\n========== SVM ==========")

svm_scaler = StandardScaler()
X_train_svm = svm_scaler.fit_transform(X_train)
X_test_svm = svm_scaler.transform(X_test)

best_svm_acc = 0
best_svm = None
best_svm_params = None

for kernel in ["linear", "rbf"]:
    for C in [1, 5, 10, 20, 50]:
        for gamma in ["scale", "auto", 0.01, 0.1]:

            model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                probability=True,
                random_state=42
            )

            model.fit(X_train_svm, y_train_encoded)
            pred = model.predict(X_test_svm)
            acc = accuracy_score(y_test_encoded, pred)

            print(f"kernel={kernel}, C={C}, gamma={gamma}, acc={acc:.4f}")

            if acc > best_svm_acc:
                best_svm_acc = acc
                best_svm = model
                best_svm_params = (kernel, C, gamma)

svm_pred = best_svm.predict(X_test_svm)
svm_metrics = get_metrics(y_test_encoded, svm_pred)

save_confusion_matrix(y_test_encoded, svm_pred, "SVM 4-Class", "svm_4class_confusion_matrix.png")

joblib.dump(best_svm, os.path.join(MODEL_FOLDER, "svm_4class_model.pkl"))
joblib.dump(svm_scaler, os.path.join(MODEL_FOLDER, "svm_4class_scaler.pkl"))

all_results.append({
    "Model": "SVM",
    "Best Parameters": str(best_svm_params),
    **svm_metrics
})


# =========================
# BALANCED SVM
# =========================

print("\n========== BALANCED SVM ==========")

best_bal_acc = 0
best_bal = None
best_bal_params = None

for kernel in ["linear", "rbf"]:
    for C in [1, 5, 10, 15, 20, 30, 50]:
        for gamma in ["scale", "auto", 0.01]:

            model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                class_weight="balanced",
                probability=True,
                random_state=42
            )

            model.fit(X_train_svm, y_train_encoded)
            pred = model.predict(X_test_svm)
            acc = accuracy_score(y_test_encoded, pred)

            print(f"balanced kernel={kernel}, C={C}, gamma={gamma}, acc={acc:.4f}")

            if acc > best_bal_acc:
                best_bal_acc = acc
                best_bal = model
                best_bal_params = (kernel, C, gamma)

bal_pred = best_bal.predict(X_test_svm)
bal_metrics = get_metrics(y_test_encoded, bal_pred)

save_confusion_matrix(y_test_encoded, bal_pred, "Balanced SVM 4-Class", "balanced_svm_4class_confusion_matrix.png")

joblib.dump(best_bal, os.path.join(MODEL_FOLDER, "balanced_svm_4class_model.pkl"))
joblib.dump(svm_scaler, os.path.join(MODEL_FOLDER, "balanced_svm_4class_scaler.pkl"))

all_results.append({
    "Model": "Balanced SVM",
    "Best Parameters": str(best_bal_params),
    **bal_metrics
})


# =========================
# EXTRA TREES
# =========================

print("\n========== EXTRA TREES ==========")

best_et_acc = 0
best_et = None
best_et_params = None

for n in [100, 200, 300, 500]:
    for depth in [None, 20, 30, 40]:
        for max_features in ["sqrt", "log2", None]:

            model = ExtraTreesClassifier(
                n_estimators=n,
                max_depth=depth,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train_encoded)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test_encoded, pred)

            print(f"Trees={n}, depth={depth}, max_features={max_features}, acc={acc:.4f}")

            if acc > best_et_acc:
                best_et_acc = acc
                best_et = model
                best_et_params = (n, depth, max_features)

et_pred = best_et.predict(X_test)
et_metrics = get_metrics(y_test_encoded, et_pred)

save_confusion_matrix(y_test_encoded, et_pred, "Extra Trees 4-Class", "extra_trees_4class_confusion_matrix.png")

joblib.dump(best_et, os.path.join(MODEL_FOLDER, "extra_trees_4class_model.pkl"))

all_results.append({
    "Model": "Extra Trees",
    "Best Parameters": str(best_et_params),
    **et_metrics
})


# =========================
# FINAL RESULTS
# =========================

final_results = pd.DataFrame(all_results)
final_results = final_results.sort_values(by="Accuracy", ascending=False)

print("\n========== FINAL 4-CLASS MODEL COMPARISON ==========")
print(final_results)

final_results.to_csv(
    os.path.join(RESULTS_FOLDER, "final_model_comparison_4class.csv"),
    index=False
)

plt.figure(figsize=(9, 5))
plt.bar(final_results["Model"], final_results["Accuracy"], color="#2563eb")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("4-Class Final Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "final_model_accuracy_comparison_4class.png"))
plt.close()

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(final_results["Model"]))
width = 0.2

plt.figure(figsize=(11, 6))

for i, metric in enumerate(metrics):
    plt.bar(x + i * width, final_results[metric], width, label=metric)

plt.xlabel("Model")
plt.ylabel("Score")
plt.title("4-Class Accuracy, Precision, Recall, and F1-score Comparison")
plt.xticks(x + width * 1.5, final_results["Model"])
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "final_model_performance_comparison_4class.png"))
plt.close()

best_model = final_results.iloc[0]

with open(os.path.join(RESULTS_FOLDER, "best_model_4class.txt"), "w") as file:
    file.write("Best 4-Class Model\n")
    file.write("==================\n")
    file.write(str(best_model))

print("\nTraining completed successfully.")
print("\nSaved 4-class models in:")
print(MODEL_FOLDER)

print("\nSaved 4-class results in:")
print(RESULTS_FOLDER)

print("\nBest 4-class model:")
print(best_model)
