"""
generate_data.py
Generates a realistic AQI training dataset using India's CPCB
(Central Pollution Control Board) breakpoint tables to compute
ground-truth AQI from pollutant concentrations.

Pollutants used: PM2.5, PM10, NO2, SO2, CO, O3
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# CPCB breakpoints: (C_low, C_high, I_low, I_high) — kept contiguous
# (each bracket's low = previous bracket's high) so there are no gaps
# a concentration value can fall through.
BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
              (90, 120, 200, 300), (120, 250, 300, 400), (250, 380, 400, 500), (380, 100000, 500, 600)],
    "PM10":  [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200),
              (250, 350, 200, 300), (350, 430, 300, 400), (430, 600, 400, 500), (600, 100000, 500, 600)],
    "NO2":   [(0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 200),
              (180, 280, 200, 300), (280, 400, 300, 400), (400, 500, 400, 500), (500, 100000, 500, 600)],
    "SO2":   [(0, 40, 0, 50), (40, 80, 50, 100), (80, 380, 100, 200),
              (380, 800, 200, 300), (800, 1600, 300, 400), (1600, 2000, 400, 500), (2000, 100000, 500, 600)],
    "CO":    [(0, 1.0, 0, 50), (1.0, 2.0, 50, 100), (2.0, 10, 100, 200),
              (10, 17, 200, 300), (17, 34, 300, 400), (34, 50, 400, 500), (50, 10000, 500, 600)],
    "O3":    [(0, 50, 0, 50), (50, 100, 50, 100), (100, 168, 100, 200),
              (168, 208, 200, 300), (208, 748, 300, 400), (748, 1000, 400, 500), (1000, 100000, 500, 600)],
}


def sub_index(pollutant: str, conc: float) -> float:
    """Compute CPCB sub-index for a single pollutant concentration."""
    bps = BREAKPOINTS[pollutant]
    conc = max(conc, 0)
    for c_low, c_high, i_low, i_high in bps:
        if c_low <= conc <= c_high:
            return (i_high - i_low) / (c_high - c_low) * (conc - c_low) + i_low
    # above the highest defined range -> cap at max index of table
    return bps[-1][3]


def compute_aqi(row: dict) -> float:
    """Overall AQI = max of individual pollutant sub-indices (CPCB rule)."""
    return max(sub_index(p, row[p]) for p in BREAKPOINTS)


def generate_dataset(n_samples: int = 6000) -> pd.DataFrame:
    """
    Simulate pollutant concentrations across a range of pollution
    scenarios (clean / moderate / heavily polluted days), then derive
    AQI from the real CPCB formula so the ML model learns a genuine
    physical/regulatory relationship, not noise.
    """
    rows = []
    for _ in range(n_samples):
        # pick a "pollution regime" so the dataset spans clean -> hazardous days
        regime = np.random.choice(["clean", "moderate", "poor", "severe"], p=[0.25, 0.35, 0.25, 0.15])
        scale = {"clean": 0.3, "moderate": 0.7, "poor": 1.3, "severe": 2.2}[regime]

        pm25 = max(0, np.random.normal(60 * scale, 20))
        pm10 = max(0, pm25 * np.random.uniform(1.3, 2.0) + np.random.normal(0, 15))
        no2 = max(0, np.random.normal(40 * scale, 15))
        so2 = max(0, np.random.normal(20 * scale, 10))
        co = max(0, np.random.normal(1.2 * scale, 0.5))
        o3 = max(0, np.random.normal(50 * scale, 20))

        row = {"PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2, "CO": co, "O3": o3}
        row["AQI"] = compute_aqi(row) + np.random.normal(0, 3)  # small sensor noise
        rows.append(row)

    df = pd.DataFrame(rows)
    df["AQI"] = df["AQI"].clip(lower=0, upper=600).round(1)
    for col in ["PM2.5", "PM10", "NO2", "SO2", "O3"]:
        df[col] = df[col].round(1)
    df["CO"] = df["CO"].round(2)
    return df


def aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


if __name__ == "__main__":
    df = generate_dataset(6000)
    df["Category"] = df["AQI"].apply(aqi_category)
    df.to_csv("aqi_dataset.csv", index=False)
    print(f"Generated {len(df)} rows -> aqi_dataset.csv")
    print(df["Category"].value_counts())
    print(df.head())
