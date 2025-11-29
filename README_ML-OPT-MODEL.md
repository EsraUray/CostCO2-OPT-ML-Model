# Cost-based CO₂ Prediction Model

This repository provides a machine learning model for predicting the
objective function **fmin(CO2Cost)** of retaining-wall design cases.

## Files

- `CostCO2MLModelFinal.py` – loads the trained model and exposes the function `predict_fmin_co2cost`.
- `PredictionModel.py` – example script showing how to call the prediction function.
- `best_co2_model.joblib` – serialized best-performing model.
- `feature_columns.json` – list of input features required by the model.

## Usage

```python
from CostCO2MLModelFinal import predict_fmin_co2cost
import pandas as pd

row = {
    "H": 4,
    "q": 0,
    "SDS": 0.84,
    "X1": 2.8,
    "X2": 0.42,
    "X3": 0.3,
    "X4": 0.3,
    "X5": 0.32,
    "X6": 0.0,
    "X7": 0.0,
    "X8": 0.0,
    "X9": 14,
    "X10": 10,
    "X11": 10,
    "X12": 12,
    "X13": 15,
    "X14": 10,
    "X15": 13,
    "X16": 10,
    "SC_ZB": False,
    "SC_ZC": False,
    "SC_ZE": False,
}

df_new = pd.DataFrame([row])
y_pred = predict_fmin_co2cost(df_new)
print("Predicted fmin(CO2Cost):", y_pred[0])
