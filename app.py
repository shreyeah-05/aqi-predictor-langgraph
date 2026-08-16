"""
app.py
Streamlit interface for the AQI Predictor.

Reuses the exact same LangGraph pipeline defined in agent.py
(validate_input -> predict_aqi -> classify_category -> explain),
just with a UI instead of a hardcoded sample reading.

Run:
    export GEMINI_API_KEY=your_key_here   (or enter it in the sidebar)
    streamlit run app.py
"""

import os
import streamlit as st
from agent import build_graph, FEATURES

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="centered")

CATEGORY_COLORS = {
    "Good": "#2ecc71",
    "Satisfactory": "#a3d977",
    "Moderate": "#f1c40f",
    "Poor": "#e67e22",
    "Very Poor": "#e74c3c",
    "Severe": "#8b0000",
}

# Reasonable default/min/max ranges per pollutant, for sensible sliders
POLLUTANT_CONFIG = {
    "PM2.5": {"unit": "µg/m³", "min": 0.0, "max": 500.0, "default": 60.0, "step": 1.0},
    "PM10":  {"unit": "µg/m³", "min": 0.0, "max": 600.0, "default": 100.0, "step": 1.0},
    "NO2":   {"unit": "µg/m³", "min": 0.0, "max": 400.0, "default": 40.0, "step": 1.0},
    "SO2":   {"unit": "µg/m³", "min": 0.0, "max": 400.0, "default": 20.0, "step": 1.0},
    "CO":    {"unit": "mg/m³", "min": 0.0, "max": 50.0,  "default": 1.0, "step": 0.1},
    "O3":    {"unit": "µg/m³", "min": 0.0, "max": 400.0, "default": 40.0, "step": 1.0},
}

# ---------------------------------------------------------------------
# Sidebar: API key + about
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    # st.secrets is the most reliable way to read Community Cloud secrets;
    # fall back to a regular env var for local runs, then to manual entry.
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        default_key = ""
    if not default_key:
        default_key = os.environ.get("GEMINI_API_KEY", "")

    api_key_input = st.text_input(
        "Gemini API key",
        value=default_key,
        type="password",
        help="Free, no card required — get one at aistudio.google.com",
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input

    st.caption(
        "✅ Key detected" if os.environ.get("GEMINI_API_KEY") else "⚠️ No key detected yet"
    )

    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "Predicts India's Air Quality Index (AQI) from pollutant readings "
        "using a RandomForest model trained on real CPCB station data "
        "(26 Indian cities, 2015-2020), then explains the result with "
        "Google Gemini.\n\n"
        "MAE: 21.9 AQI points · R²: 0.90"
    )

# ---------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------
st.title("🌫️ AQI Predictor")
st.caption("Enter pollutant readings to predict India's Air Quality Index (CPCB scale).")

st.subheader("Pollutant readings")
cols = st.columns(2)
pollutants = {}
for i, feat in enumerate(FEATURES):
    cfg = POLLUTANT_CONFIG[feat]
    with cols[i % 2]:
        pollutants[feat] = st.number_input(
            f"{feat} ({cfg['unit']})",
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["default"],
            step=cfg["step"],
        )

predict_clicked = st.button("Predict AQI", type="primary", use_container_width=True)

if predict_clicked:
    if not os.environ.get("GEMINI_API_KEY"):
        st.warning(
            "No Gemini API key set — you'll still get the AQI prediction, but "
            "the plain-language explanation step will be skipped. Add a free "
            "key in the sidebar to enable it."
        )

    with st.spinner("Running prediction..."):
        app_graph = build_graph()
        result = app_graph.invoke({"pollutants": pollutants})

    if result.get("error"):
        st.error(result["error"])
    else:
        aqi = result["predicted_aqi"]
        category = result["category"]
        color = CATEGORY_COLORS.get(category, "#999999")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted AQI", aqi)
        with col2:
            st.markdown(
                f"<div style='padding:10px; border-radius:8px; background-color:{color}; "
                f"color:white; text-align:center; font-weight:600; font-size:1.1em;'>"
                f"{category}</div>",
                unsafe_allow_html=True,
            )

        st.subheader("What this means")
        explanation = result.get("explanation", "")
        if explanation and "Could not generate" not in explanation:
            st.write(explanation)
        else:
            st.warning(explanation or "Explanation unavailable for an unknown reason.")
