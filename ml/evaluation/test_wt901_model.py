import joblib

print("Loading model...")

model = joblib.load("svm_4class_model.pkl")
scaler = joblib.load("svm_4class_scaler.pkl")

print("Model loaded successfully!")
print(model)
print(scaler)
