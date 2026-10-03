# Review Demonstration Guide

## What to show the review panel

1. Open the Streamlit dashboard and show Overview, EDA, Model Comparison, Explainable AI, Recommendations, and YOLO Human Detection tabs.
2. Explain that 70,902 cleaned energy records (39 homes, five years) were chronologically split into train/validation/test data.
3. Show the six-model comparison. Gradient Boosting is final because it has the best test RMSE (8.021 kWh) and R² (0.620).
4. Show the SHAP figure and state: it explains predictive associations, not causal relationships.
5. Select a house in Recommendations and explain forecast, 7-day baseline, status, recommendation, and estimated saving.
6. Show YOLO test results: precision 0.939, recall 0.907, mAP50 0.962, mAP50-95 0.577 on 75 test images.
7. Explain that energy records and images are separate datasets. They can be operationally combined only when home/camera ID and timestamp are available.

## Likely viva questions

- **Why Gradient Boosting?** Lowest held-out test RMSE and highest R² among all evaluated models.
- **Why not deploy the hybrid?** It was tuned on validation data but did not beat Gradient Boosting on the independent test set.
- **Why is MAPE high?** Some actual kWh values are close to zero, making percentage errors unstable. MAE/RMSE are primary metrics.
- **Does YOLO prove occupancy?** No. It provides a recent person-detection signal; it must be time-matched before use in automation.
- **Are savings real?** No. The dashboard reports a transparent rule-based estimate; field measurement is future work.
