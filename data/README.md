# Dataset Organisation

This project deliberately keeps the source and derived datasets separate. Never overwrite a raw file.

```text
data/
├── raw/
│   └── energy_5year_raw.csv          # Original source file: 71,214 rows
├── processed/
│   ├── energy_model_data.csv         # Cleaned/engineered energy data: 70,902 rows
│   └── data_profile.json             # Cleaning audit and date/split summary
└── yolo/
    ├── raw/                          # Downloaded original person-detection images/labels
    ├── processed/                    # Validated YOLO train/val/test dataset
    └── reports/                      # YOLO metrics, plots and predictions
```

## Energy dataset lineage

`raw/energy_5year_raw.csv` → clean missing values and duplicates → filter latest five years → create calendar and lag features → `processed/energy_model_data.csv`.

The processed file includes the original useful fields plus engineered `kwh_lag_1`, `kwh_lag_7`, `kwh_rolling_7`, calendar fields and the next-day prediction target `target_next_day_kwh`.

## YOLO dataset lineage

Downloaded images and annotations will first be preserved in `yolo/raw`. A validation script will then check image/label pairs, invalid bounding boxes, duplicate files, class name and split balance before writing the approved dataset under `yolo/processed`.
