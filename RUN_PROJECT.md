# Run the Complete Project

Open Command Prompt in `D:\final_miniproject` and run the commands below.

## Review mode (uses completed outputs; no retraining)

```cmd
.\.venv\Scripts\python.exe src\export_dashboard_data.py
.\.venv\Scripts\python.exe src\predict_yolo.py
.\.venv\Scripts\python.exe src\generate_future_forecasts.py
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

Open `http://localhost:8501`. Keep the Command Prompt open while the dashboard is running.

In the `Live Energy + YOLO` tab, select a house, upload a current room/CCTV image, and click **Detect human and generate recommendation**. The combined event log is saved to `reports/integration/occupancy_energy_events.csv`.

## Full reproducibility mode

```cmd
.\.venv\Scripts\python.exe src\prepare_data.py --input "data\raw\energy_5year_raw.csv"
.\.venv\Scripts\python.exe src\eda.py
.\.venv\Scripts\python.exe src\train_ml.py
.\.venv\Scripts\python.exe src\train_dl.py --epochs 35
.\.venv\Scripts\python.exe src\train_hybrid.py
.\.venv\Scripts\python.exe src\compare_models.py
.\.venv\Scripts\python.exe src\explain_and_recommend.py
.\.venv\Scripts\python.exe src\export_dashboard_data.py
.\.venv\Scripts\python.exe src\generate_future_forecasts.py
```

YOLO is already trained. To test its saved model without training again:

```cmd
.\.venv\Scripts\python.exe src\train_yolo.py --evaluate-only
.\.venv\Scripts\python.exe src\predict_yolo.py
```

Do not rerun YOLO training before a review unless you intentionally want a new experiment; it takes about 2.5 hours on CPU.
