# Project File Guide – Tamil

## Main folders

```text
final_miniproject/
├── app/                 # Streamlit dashboard code
├── src/                 # All Python data/model/training code
├── data/                # Original and cleaned datasets
├── models/              # Saved trained models only
├── reports/             # Generated charts, CSV results and logs
├── docs/                # Documentation only; no executable code
├── START_REVIEW_DASHBOARD.cmd
├── RUN_YOLO_DEMO.cmd
├── RUN_PROJECT.md
└── README.md
```

## Coding files (`src/`)

| File | Purpose | Output |
|---|---|---|
| `prepare_data.py` | Cleans original energy data; creates lags, calendar features and next-day target | `data/processed/energy_model_data.csv` |
| `eda.py` | Daily/monthly/correlation EDA charts | `reports/eda_*.png` |
| `train_ml.py` | Random Forest, XGBoost, Gradient Boosting | ML models, predictions, `ml_metrics.json` |
| `train_dl.py` | LSTM and GRU training | DL models, predictions, `dl_metrics.json` |
| `train_hybrid.py` | GB + LSTM weighted hybrid validation experiment | hybrid metrics and predictions |
| `compare_models.py` | Compares all ML/DL/Hybrid models | `five_model_comparison.png` |
| `explain_and_recommend.py` | SHAP/XAI, permutation importance, energy recommendations | SHAP chart, importance CSV, recommendations |
| `generate_future_forecasts.py` | Latest-data next-day forecast for all 39 homes | `future_energy_forecasts.csv` |
| `prepare_yolo_data.py` | Validates YOLO images/labels and creates 500-image split | `data/yolo/processed/` |
| `train_yolo.py` | YOLOv8 training or test evaluation | YOLO curves, best model, test metrics |
| `predict_yolo.py` | Runs person detection on images | prediction images and event CSV |
| `export_dashboard_data.py` | Creates Power BI import tables | `reports/powerbi/` |

## Dashboard code (`app/`)

`streamlit_app.py` is the only dashboard code file. It contains:

- Energy overview and actual-vs-predicted chart
- Daily, monthly, yearly, seasonal EDA
- Six-model comparison
- SHAP Explainable AI
- House-wise energy recommendation
- Live Energy + YOLO image/camera workflow
- YOLO accuracy, PR curve and prediction images

## Data files (`data/`) – no code

| Folder | Contains |
|---|---|
| `data/raw/` | Original energy CSV. Never edit this file. |
| `data/processed/` | Cleaned energy dataset and data profile. |
| `data/yolo/raw/` | Roboflow downloaded original YOLO dataset. |
| `data/yolo/processed/` | Validated 500-image train/val/test YOLO dataset. |
| `data/yolo/reports/` | YOLO test charts, confusion matrix and training outputs. |

## Models (`models/`) – no code

Saved artifacts used for prediction: Gradient Boosting, RF, XGBoost, LSTM, GRU and `yolo_person_detector.pt`. Do not edit them manually.

## Reports (`reports/`) – no code

Generated outputs only:

- `eda_*.png`: EDA charts
- `five_model_comparison.png`: final model graph
- `shap_global_importance.png`: Explainable AI
- `recommendations.csv`: prediction, status, saving estimate and recommendation
- `future_energy_forecasts.csv`: next-day forecast for 39 homes
- `integration/occupancy_energy_events.csv`: live Energy + YOLO combined events
- `powerbi/`: Power BI import CSV files

## How to run for review

Double-click `START_REVIEW_DASHBOARD.cmd`. Then open `http://localhost:8501`.

Do **not** run every training script before a review. Training is only for reproducing the research; it takes time. The saved models and outputs are already ready.
