"""
agent.py
A LangGraph agent that wraps the trained AQI ML model.

Graph flow:
  validate_input -> predict_aqi -> classify_category -> explain -> END
  (on invalid input, later nodes just pass the error through to explain)

Run:
    export GEMINI_API_KEY=your_key_here   (get one free, no card, at aistudio.google.com)
    python3 agent.py
"""

import os
import joblib
import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from google import genai

MODEL_PATH = "aqi_model.joblib"
DATA_PATH = "aqi_dataset_real.csv"
TRAIN_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]


class AQIState(TypedDict):
    pollutants: dict
    error: Optional[str]
    predicted_aqi: Optional[float]
    category: Optional[str]
    explanation: Optional[str]


def _load_or_train_model():
    """
    Loads the saved model if present. If not (e.g. on a fresh deploy
    where the ~57MB .joblib file wasn't pushed to GitHub), trains one
    on the fly from aqi_dataset_real.csv -- takes a few seconds, and
    means the app never depends on a large binary being in the repo.
    """
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], bundle["features"]

    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_csv(DATA_PATH)
    X = df[TRAIN_FEATURES]
    y = df["AQI"]

    model = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y)

    try:
        joblib.dump({"model": model, "features": TRAIN_FEATURES}, MODEL_PATH)
    except OSError:
        pass  # read-only filesystem on some deploy platforms -- fine to skip

    return model, TRAIN_FEATURES


MODEL, FEATURES = _load_or_train_model()

CATEGORY_RANGES = [
    (0, 50, "Good"),
    (50, 100, "Satisfactory"),
    (100, 200, "Moderate"),
    (200, 300, "Poor"),
    (300, 400, "Very Poor"),
    (400, 600, "Severe"),
]


def classify(aqi: float) -> str:
    for low, high, label in CATEGORY_RANGES:
        if low <= aqi <= high:
            return label
    return "Severe"


def validate_input(state: AQIState) -> AQIState:
    pollutants = state.get("pollutants", {})
    missing = [f for f in FEATURES if f not in pollutants]
    if missing:
        return {**state, "error": f"Missing required pollutant readings: {missing}"}
    for f in FEATURES:
        val = pollutants[f]
        if not isinstance(val, (int, float)) or val < 0:
            return {**state, "error": f"Invalid value for {f}: {val} (must be a non-negative number)"}
    return {**state, "error": None}


def predict_aqi(state: AQIState) -> AQIState:
    if state.get("error"):
        return state
    row = pd.DataFrame([[state["pollutants"][f] for f in FEATURES]], columns=FEATURES)
    pred = float(MODEL.predict(row)[0])
    return {**state, "predicted_aqi": round(pred, 1)}


def classify_category(state: AQIState) -> AQIState:
    if state.get("error"):
        return state
    return {**state, "category": classify(state["predicted_aqi"])}


def explain(state: AQIState) -> AQIState:
    if state.get("error"):
        return {**state, "explanation": f"Could not generate a prediction: {state['error']}"}

    client = genai.Client()
    pollutant_summary = ", ".join(f"{k}: {v}" for k, v in state["pollutants"].items())

    prompt = (
        f"Air quality readings: {pollutant_summary}. "
        f"Predicted AQI: {state['predicted_aqi']} ({state['category']} category, India CPCB scale). "
        "In 3-4 short sentences: (1) state what this AQI level means for a general "
        "healthy adult, (2) name the pollutant most likely driving this reading, "
        "and (3) give one practical precaution. Keep it plain-language, no jargon, "
        "no markdown headers."
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        explanation = response.text
    except Exception as e:
        # Covers missing/invalid API key, rate limits, and transient
        # server-side issues (e.g. google.genai.errors.ServerError) --
        # the app should still show the AQI prediction even if the
        # explanation step fails.
        explanation = f"Could not generate explanation: {e}"

    return {**state, "explanation": explanation}


def build_graph():
    graph = StateGraph(AQIState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("predict_aqi", predict_aqi)
    graph.add_node("classify_category", classify_category)
    graph.add_node("explain", explain)

    graph.set_entry_point("validate_input")
    graph.add_edge("validate_input", "predict_aqi")
    graph.add_edge("predict_aqi", "classify_category")
    graph.add_edge("classify_category", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    sample_reading = {
        "PM2.5": 145.0, "PM10": 210.0, "NO2": 55.0,
        "SO2": 18.0, "CO": 1.8, "O3": 60.0,
    }
    result = app.invoke({"pollutants": sample_reading})
    print("Input pollutants :", sample_reading)
    print("Predicted AQI    :", result["predicted_aqi"])
    print("Category         :", result["category"])
    print("Explanation      :", result["explanation"])
