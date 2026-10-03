# Two-Dataset Professional Architecture

## 1. Energy Forecasting Dataset

**Input:** household energy, weather, building and appliance fields.

**Pipeline:** raw CSV → cleaning → EDA → feature engineering → RF/XGBoost/GB/LSTM/GRU/Hybrid evaluation → Gradient Boosting forecast → SHAP/XAI → energy status and saving estimate.

**Standalone outputs:** EDA charts, test metrics, model comparison graph, prediction file, SHAP chart, energy dashboard and Power BI dashboard.

## 2. Human Detection Dataset

**Input:** labelled images containing the single class `person`.

**Pipeline:** raw images/labels → image/label integrity validation → duplicate and corrupt-image removal → fixed 70/15/15 train/validation/test split → YOLOv8 training → test precision, recall, mAP50 and mAP50-95 → person-present inference.

**Standalone outputs:** dataset audit report, class/split chart, YOLO training curves, confusion matrix, validation predictions, test metrics, detection dashboard.

## 3. Unified Recommendation Layer

The two source datasets are **not merged row-by-row** because they have different units and time granularities. At deployment, they are joined only through a shared `home_or_camera_id` and timestamp window:

```text
Energy forecast for a home/day + latest YOLO person_present event
→ occupancy-aware status
→ safe recommendation and estimated energy saving
```

If a reliable common ID and time window are unavailable, the dashboard will display both outputs together but will not claim that they were statistically joined.
