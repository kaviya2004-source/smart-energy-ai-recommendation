AI Smart Energy Management – Final Demo Package

Files:
1. streamlit_app.py
2. demo_energy_dataset_1.csv
3. demo_energy_dataset_2.csv
4. demo_energy_dataset_3.csv
5. demo_energy_dataset_4.csv
6. demo_energy_dataset_5.csv

Each demo dataset contains exactly 500 records (5 houses × 100 days).
Required fields for upload are timestamp and kwh; house_id and other energy/weather
fields are included to make the demo realistic.

Use:
- Replace your existing app/streamlit_app.py with the final file after keeping a backup.
- Keep your existing project folders (models, reports, data, src) unchanged.
- Run from the project root:
  .\.venv\Scripts\activate
  streamlit run app\streamlit_app.py

The dashboard expects the existing project model/report files for the full project
modules. The upload demo validates energy files and requires at least 500 records.
