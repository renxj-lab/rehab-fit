import joblib

model = joblib.load("wt901_svm_model.pkl")
scaler = joblib.load("wt901_svm_scaler.pkl")
features = joblib.load("wt901_feature_columns.pkl")

print("WT901 model loaded successfully!")
print(model)
print(scaler)
print("Number of features:", len(features))
print(features[:10])
