"""
train_model.py
Trains a RandomForestRegressor to predict AQI from pollutant
concentrations, evaluates it, and saves the model + feature list
for use by the LangGraph agent (agent.py).
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET = "AQI"


def main():
    df = pd.read_csv("aqi_dataset.csv")

    X = df[FEATURES]
    y = df[TARGET]

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

    print("=== Model Evaluation ===")
    print(f"MAE  : {mae:.2f} AQI points")
    print(f"RMSE : {rmse:.2f} AQI points")
    print(f"R^2  : {r2:.4f}")

    print("\n=== Feature Importance ===")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"{feat:8s}: {imp:.3f}")

    joblib.dump({"model": model, "features": FEATURES}, "aqi_model.joblib")
    print("\nSaved model -> aqi_model.joblib")


if __name__ == "__main__":
    main()
