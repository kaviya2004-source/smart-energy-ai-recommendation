# Power BI Dashboard Specification

Import `reports/all_model_comparison.csv`, `reports/predictions_*.csv`, `sreports/permutation_importance.csv`, and `reports/recommendations.csv` after the scripts run.

Create four pages:

1. **Executive overview:** best model card (lowest RMSE), mean actual/predicted kWh cards, line chart of actual vs predicted daily kWh, and a house/date slicer.
2. **Model comparison:** clustered bars for RMSE, MAE, MAPE, and R² across all five algorithms. Add a text box stating that all numbers are chronological held-out test metrics.
3. **Explainability:** bar chart of top permutation-importance features and a table of prediction, actual, absolute error, house, and date.
4. **Savings actions:** predicted kWh trend, recommendation table, high-consumption count, and an occupancy/person-present status tile when YOLO events are available.

## Required Power BI slicers

Add these slicers at the top of the relevant pages:

1. `house_id` — lets a reviewer analyse one household.
2. `timestamp` / date — use a Between date slicer for a period.
3. `energy_status` — High, Normal, Low forecast status.
4. `Model` — use only on the Model Comparison page.
5. `season` and `year` — use on EDA pages.

Use *Edit interactions* so the house/date slicers filter charts and tables on the same page. Do not use YOLO metrics as an energy-data filter because YOLO images have no shared timestamp/home ID.

Recommended DAX measures:

```DAX
Absolute Error = ABS(AVERAGE(Predictions[actual]) - AVERAGE(Predictions[prediction]))
Daily kWh Actual = AVERAGE(Predictions[actual])
Daily kWh Predicted = AVERAGE(Predictions[prediction])
High Forecast Days = COUNTROWS(FILTER(Recommendations, Recommendations[prediction] >= 30))
```

Keep the prediction file for the selected final model as the default report table. Include the model name and run date in the report title so results remain reproducible.
