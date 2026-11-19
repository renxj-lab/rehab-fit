# Rehab Fit – Model Evaluation

## 1. Overview

Four machine learning classification algorithms were evaluated for recognising shoulder physiotherapy movements using wearable IMU sensor data.

## 2. Model Performance

| Model | Accuracy | F1-Score |
|---|---:|---:|
| Linear SVM (C=5) | 96.30% | 96.30% |
| KNN (k=13) | 94.21% | 94.21% |
| Random Forest | 92.19% | 92.11% |
| Extra Trees | 91.21% | 91.10% |

## 3. Best Performing Model

The Linear Support Vector Machine (SVM), configured with C=5, achieved the highest reported accuracy of approximately 96.30%.

## 4. Model Selection Rationale

The Linear Support Vector Machine was selected as the final model because it achieved the highest accuracy and F1-score among the evaluated classifiers.

Its performance indicates that the extracted sensor features provide useful information for distinguishing between the different physiotherapy movement categories.

The final model was configured with a regularisation parameter of C=5.

Further testing with unseen users and real-world sensor conditions is required to evaluate generalisation.

## 5. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

## 6. Conclusion

Among the evaluated models, Linear SVM demonstrated the strongest overall accuracy and was selected as the final classification model based on the reported evaluation results.

Further validation using additional participants and real-world testing is needed to assess generalisation.
