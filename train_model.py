"""
train_model.py
Trains a RandomForestRegressor to predict AQI from pollutant
concentrations, using REAL CPCB-sourced data from 26 Indian cities
(2015-2020). Evaluates the model, runs SHAP analysis for feature
importance, and saves the model for use by the LangGraph agent
(agent.py).
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  # no display in this environment
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET = "AQI"
DATA_FILE = "aqi_dataset_real.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    # 80% train / 20% unseen test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("=== Model Evaluation (real CPCB data, 26 Indian cities, 2015-2020) ===")
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"MAE  : {mae:.2f} AQI points")
    print(f"RMSE : {rmse:.2f} AQI points")
    print(f"R^2  : {r2:.4f}")

    print("\n=== Built-in Feature Importance (Gini-based) ===")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"{feat:8s}: {imp:.3f}")

    # ---------------- SHAP analysis ----------------
    print("\nRunning SHAP analysis (this samples a subset of the test set)...")
    explainer = shap.TreeExplainer(model)
    # SHAP on the full test set can be slow for large forests; sample for speed
    sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = sorted(
        zip(FEATURES, mean_abs_shap), key=lambda x: -x[1]
    )
    print("\n=== SHAP Feature Importance (mean |SHAP value|) ===")
    for feat, val in shap_importance:
        print(f"{feat:8s}: {val:.2f}")

    # Save SHAP summary bar plot
    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png", dpi=150)
    plt.close()
    print("\nSaved SHAP plot -> shap_feature_importance.png")

    joblib.dump({"model": model, "features": FEATURES}, "aqi_model.joblib")
    print("Saved model -> aqi_model.joblib")


if __name__ == "__main__":
    main()
