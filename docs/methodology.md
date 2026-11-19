# Rehab Fit – Methodology

## 1. Data Collection
Motion data is collected using a wearable WT901BLE 9-axis IMU sensor.

## 2. Data Preprocessing
Collected sensor readings are prepared for machine learning through data cleaning, segmentation and preprocessing.

## 3. Feature Extraction
A set of 104 features is extracted from segmented motion windows to represent movement characteristics.

## 4. Model Training
Multiple classification algorithms are evaluated, including:
- Linear Support Vector Machine
- K-Nearest Neighbours
- Random Forest
- Extra Trees

## 5. Evaluation
The models are compared using classification accuracy and F1-score.

## 6. Application Integration
The selected machine learning model is integrated into the application workflow to support exercise recognition and monitoring.
