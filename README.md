# AQI Predictor (ML + LangGraph)

Predicts India's Air Quality Index (AQI) from pollutant readings using a
trained ML model, wrapped in a LangGraph agent that also explains the
result in plain language. Also includes a Streamlit web interface.

## Data source (real, not synthetic)

Training data is `city_day.csv` — real day-level air quality readings
from **26 Indian cities, 2015–2020**, sourced from CPCB (Central
Pollution Control Board) monitoring stations. This is the widely-used
"Air Quality Data in India" dataset originally published on Kaggle by
Rohan Rao: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

## Pipeline

| File | Purpose |
|---|---|
| `real_city_day.csv` | Raw real dataset: 29,531 rows, 26 cities, 2015–2020, with real missing values. |
| `prepare_real_data.py` | Cleans the raw data: drops rows with no AQI label, imputes missing pollutant readings using each city's own median (not a single global median, since baseline pollution varies hugely between e.g. Delhi and Aizawl). Outputs `aqi_dataset_real.csv`. |
| `train_model.py` | Trains a `RandomForestRegressor` on `PM2.5, PM10, NO2, SO2, CO, O3 -> AQI` using an 80/20 train/test split. Evaluates with MAE/RMSE/R², computes both built-in feature importance and **SHAP** analysis, and saves `shap_feature_importance.png`. |
| `agent.py` | LangGraph agent: `validate_input -> predict_aqi -> classify_category -> explain`. The `explain` node calls Google's Gemini API (free tier) for a plain-language health summary. |
| `app.py` | Streamlit web interface — enter pollutant readings, get the predicted AQI, category, and explanation. |
| `requirements.txt` | Dependency list for local runs or deployment. |

## Current model performance (on real data)

- **MAE**: ~21.9 AQI points
- **R²**: ~0.90

This is meaningfully less "perfect" than a synthetic-data version would
show, and that's expected and correct — real sensor data has noise,
missing-value imputation, and genuine city-to-city variation that a
formula-generated dataset doesn't. These numbers are in the same range
as other published models trained on this same real dataset.

## Feature importance (SHAP)

SHAP (SHapley Additive exPlanations) analysis on the test set shows
**PM2.5, CO, and PM10** as the dominant predictors, with NO2, SO2, and
O3 contributing much less. See `shap_feature_importance.png` for the
plot. This mirrors what CPCB and independent studies generally find
for Indian cities — particulate matter and CO are typically the
biggest drivers of poor air quality.

## A note on preprocessing choices

The pipeline applies real missing-value imputation (per-city median)
and a genuine 80/20 train/test split. Feature **standardization** was
deliberately **not** applied to the final model: `RandomForestRegressor`
is a tree-based method and is scale-invariant (it splits on ordering,
not magnitude), so standardizing pollutant units before training would
add a preprocessing step without changing predictions. This is a
considered choice, not an oversight — worth mentioning if asked, since
it shows the reasoning rather than just following a checklist.

## What's still synthetic (being upfront about scope)

Temperature, humidity, wind speed, and pressure are **not** included
as features. Real historical meteorological data for these 26 cities
exists (e.g. via IMD or NOAA), but merging it in reliably was outside
this project's current scope. The model currently relies on the six
pollutant readings only. This would be a legitimate next step if
extending the project further.


## Setup

```bash
pip install -r requirements.txt
```

## Run it

```bash
# 1. Clean the real dataset (only needed once)
python3 prepare_real_data.py

# 2. Train the model (includes SHAP analysis)
python3 train_model.py

# 3a. Run the agent from the terminal
# Windows: set GEMINI_API_KEY=your_key_here
# Mac/Linux: export GEMINI_API_KEY=your_key_here
python3 agent.py

# 3b. OR run the web interface
streamlit run app.py
```

Get a free Gemini API key (no credit card) at https://aistudio.google.com

## Next steps you could add

- Merge in real meteorological data (temperature, humidity, wind speed,
  pressure) from IMD or NOAA historical records
- Add a `fetch_live_readings` node so the agent predicts *today's* AQI
  from a live API instead of manually entered values
- Try a gradient boosting model (XGBoost/LightGBM) for comparison
  against the RandomForest baseline
