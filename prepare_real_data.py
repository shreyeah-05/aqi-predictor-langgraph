"""
prepare_real_data.py
Cleans the real India city_day.csv dataset (26 cities, 2015-2020,
sourced from CPCB monitoring stations via Kaggle: "Air Quality Data
in India" by Rohan Rao) into a model-ready dataset.

Source: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
(mirrored copy used here: raw.githubusercontent.com — same underlying
CPCB-sourced data, since Kaggle itself isn't directly downloadable
without API credentials)
"""

import pandas as pd

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
RAW_FILE = "real_city_day.csv"
OUTPUT_FILE = "aqi_dataset_real.csv"


def main():
    df = pd.read_csv(RAW_FILE, parse_dates=["Date"])
    print(f"Raw rows: {len(df)}, cities: {df['City'].nunique()}, "
          f"date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    # Keep only the columns our model actually uses, plus AQI (target),
    # City and Date (kept for reference/EDA, not used as model features)
    df = df[["City", "Date"] + FEATURES + ["AQI"]].copy()

    # Drop rows with no AQI label at all -- can't train or evaluate on these
    before = len(df)
    df = df.dropna(subset=["AQI"])
    print(f"Dropped {before - len(df)} rows with missing AQI "
          f"({len(df)} rows remain)")

    # Missing-value imputation: fill missing pollutant readings with the
    # median for that SAME CITY (a city's typical pollution baseline is a
    # far better estimate than a single global median across 26 cities
    # with very different baseline air quality)
    for col in FEATURES:
        df[col] = df.groupby("City")[col].transform(lambda s: s.fillna(s.median()))

    # A handful of rows may still be missing a value if a city had NO
    # readings at all for that pollutant -- fall back to the global median
    for col in FEATURES:
        df[col] = df[col].fillna(df[col].median())

    missing_after = df[FEATURES].isnull().sum().sum()
    print(f"Missing pollutant values after imputation: {missing_after}")

    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned dataset -> {OUTPUT_FILE} ({len(df)} rows)")
    print(df[FEATURES + ["AQI"]].describe())


if __name__ == "__main__":
    main()
