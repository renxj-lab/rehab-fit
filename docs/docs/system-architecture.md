# Rehab Fit – System Architecture

## 1. Overview

Rehab Fit combines wearable motion sensing, machine learning and a web-based application to support shoulder physiotherapy exercise monitoring.

## 2. System Components

### Wearable Sensor Layer
- WT901BLE 9-axis IMU sensor
- Motion data acquisition
- Bluetooth Low Energy communication

### Data Processing Layer
- Sensor data preprocessing
- Window segmentation
- Feature extraction
- 104 extracted features

### Machine Learning Layer
- Linear SVM classification
- Exercise movement recognition
- Movement category prediction

### Backend Layer
- Python
- FastAPI
- Prediction and application communication

### Frontend Layer
- React
- Exercise monitoring interface
- Visual presentation of exercise information

## 3. System Workflow

1. The wearable sensor collects movement data.
2. Data is transmitted through Bluetooth Low Energy.
3. Sensor readings are processed into prediction windows.
4. Features are extracted from the sensor data.
5. The trained machine learning model classifies the movement.
6. The application presents the prediction and relevant feedback.

## 4. Future Development

- Improve real-time prediction performance.
- Enhance feedback and repetition monitoring.
- Expand testing across different users.
- Improve system reliability and usability.
