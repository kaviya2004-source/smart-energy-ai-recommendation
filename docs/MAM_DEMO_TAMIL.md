# Mam Review Demo – Tamil Steps

## Dashboard open செய்வது

1. `D:\final_miniproject` folder open செய்யவும்.
2. `START_REVIEW_DASHBOARD.cmd` file-ஐ double-click செய்யவும்.
3. Command Prompt-ல் `Local URL: http://localhost:8501` வந்ததும் browser-ல் அந்த URL open செய்யவும்.
4. Command Prompt-ஐ close செய்யக்கூடாது.

## Mam-க்கு காட்ட வேண்டிய order

1. **Overview:** final model Gradient Boosting, RMSE, R², actual vs predicted chart.
2. **EDA:** daily, monthly, yearly, seasonal energy consumption.
3. **Model Comparison:** 3 ML + 2 DL + Hybrid. Gradient Boosting best என்று காட்டவும்.
4. **Explainable AI:** SHAP feature importance மற்றும் top factors.
5. **Recommendations:** house select செய்து predicted kWh, baseline, energy status, saving recommendation காட்டவும்.
6. **YOLO Human Detection:** Precision 0.939, Recall 0.907, mAP50 0.962 மற்றும் sample detected images காட்டவும்.
7. **Live Energy + YOLO:** house select செய்யவும்; current room/CCTV photo upload செய்யவும்; `Detect human and generate recommendation` click செய்யவும். Image-ல் person count, energy forecast மற்றும் combined recommendation வரும். Event `reports/integration/occupancy_energy_events.csv`-ல் save ஆகும்.

## Viva answer

Energy prediction மற்றும் person detection datasets different. Historical dataset-ல் common camera/timestamp இல்லை. அதனால் dashboard-ல் selected house forecast மற்றும் current uploaded image detection result-ஐ operationally combine செய்கிறோம். Real deployment-ல் home/camera ID மற்றும் timestamp கிடைத்தால் இது automatic ஆகும்.
