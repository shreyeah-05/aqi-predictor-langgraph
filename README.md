# AQI Predictor (ML + LangGraph)

Predicts India's Air Quality Index (AQI) from pollutant readings using a
trained ML model, wrapped in a LangGraph agent that also explains the
result in plain language.

## How it works

| File | Purpose |
|---|---|
| `generate_data.py` | Builds a training dataset using India's real CPCB breakpoint tables (the official formula used to compute AQI from pollutant concentrations). Ground truth AQI is calculated from the formula, not guessed — so the model learns a real regulatory relationship. |
| `train_model.py` | Trains a `RandomForestRegressor` on `PM2.5, PM10, NO2, SO2, CO, O3 -> AQI`. Prints MAE / RMSE / R² and feature importances, saves `aqi_model.joblib`. |
| `agent.py` | LangGraph agent: `validate_input -> predict_aqi -> classify_category -> explain`. The `explain` node calls Google's Gemini API (free tier, no credit card) to turn the raw prediction into a short, human-readable health summary. |

## Current model performance

- **MAE**: ~2.8 AQI points
- **R²**: ~0.998

(PM2.5 dominates feature importance — this matches real-world CPCB behavior, since PM2.5 is usually the pollutant that drives the overall AQI in Indian cities.)

## Setup

```bash
pip install scikit-learn pandas numpy joblib langgraph langchain-core google-genai
```

## Get a free Gemini API key (no credit card required)

1. Go to **aistudio.google.com**
2. Sign in with a Google account
3. Click **Get API Key** → **Create API Key**
4. Copy it

Free tier: 15 requests/minute, 1,500 requests/day on `gemini-3.5-flash` — more than enough for this project.

## Run it

```bash
# 1. Generate data (only needed once, or to regenerate)
python3 generate_data.py

# 2. Train the model
python3 train_model.py

# 3. Run the agent
# Windows (Command Prompt):
set GEMINI_API_KEY=your_key_here
# Mac/Linux:
export GEMINI_API_KEY=your_key_here

python3 agent.py
```

## Example output

```
Input pollutants : {'PM2.5': 145.0, 'PM10': 210.0, 'NO2': 55.0, 'SO2': 18.0, 'CO': 1.8, 'O3': 60.0}
Predicted AQI    : 318.3
Category         : Very Poor
Explanation      : <Claude-generated plain-language summary>
```

## Using your own real-world data instead of synthetic data

For a stronger portfolio project, swap `aqi_dataset.csv` for a real
dataset — e.g. Kaggle's "Air Quality Data in India" dataset, or the
CPCB's own public station data (https://cpcb.nic.in) — as long as it
has columns for `PM2.5, PM10, NO2, SO2, CO, O3` and an `AQI` value.
`train_model.py` will work unchanged.

## Next steps you could add

- Swap in real historical data for a specific city (e.g. Thiruvananthapuram/Kochi) and note seasonal patterns
- Add a `fetch_live_readings` node that pulls current pollutant data from a live API (e.g. OpenWeather Air Pollution API) so the agent predicts *today's* AQI, not just manually entered values
- Add a Streamlit or Gradio front-end so it's demo-able for interviews
- Extend the graph with a loop: if `predicted_aqi` looks like an outlier vs. recent history, re-query for confirmation before explaining
