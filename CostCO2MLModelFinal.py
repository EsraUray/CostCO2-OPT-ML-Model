import os
import json
import joblib
import pandas as pd

BASE_DIR = r"C:\Users\User\PycharmProjects\CostCO2ModelML\deploy_model"

MODEL_PATH = os.path.join(BASE_DIR, "best_co2_model.joblib")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_columns.json")
TARGET_COL = "fmin(CO2Cost)"

# Load the model and feature list once when the module is imported
best_model = joblib.load(MODEL_PATH)
with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)


def predict_fmin_co2cost(df_input: pd.DataFrame):
    """
    df_input: DataFrame containing the columns listed in feature_columns.json
    """
    X = df_input[feature_cols].copy()
    y_pred = best_model.predict(X)
    return y_pred


if __name__ == "__main__":
    print(
        "This module is intended to be used as a function.\n"
        "Call predict_fmin_co2cost with sample data to test it."
    )
