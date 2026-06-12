# Air Quality Prediction

A machine learning pipeline that predicts **NO₂ (nitrogen dioxide) concentration** from real sensor data and fires alerts when predicted levels exceed the WHO safety threshold.

**[Live Dashboard →](https://air-quality-prediction-pzaw9n7hxlbxn8hxsth86j.streamlit.app/)**

**Dataset:** UCI Air Quality — hourly sensor readings from Perugia, Italy (March 2004 – April 2005)  
**Model:** Random Forest Regressor | **Test R²: 0.5977** | **RMSE: 34.19 µg/m³**

---

## Overview

Nitrogen dioxide is a primary indicator of urban air pollution, produced mainly by vehicle exhaust and combustion. This project builds an end-to-end NO₂ forecasting pipeline:

- Loads and parses real multi-sensor time-series data (7,700+ hourly readings)
- Cleans sentinel-encoded missing values (-200) and imputes with column medians — 81% → 100% completeness
- Engineers 13 features: 10 sensor readings + 3 temporal (hour of day, rush-hour flag, day of week)
- Trains a Random Forest ensemble inside a scikit-learn Pipeline with StandardScaler
- Evaluates with a chronological 80/20 split — no data leakage
- Flags hourly readings where predicted NO₂ exceeds the **200 µg/m³ WHO guideline** (34 breach periods detected)
- Visualises hourly and monthly NO₂ trends via an interactive Streamlit dashboard

---

## Live Demo

**[https://air-quality-prediction-pzaw9n7hxlbxn8hxsth86j.streamlit.app/](https://air-quality-prediction-pzaw9n7hxlbxn8hxsth86j.streamlit.app/)**

4 tabs:
- **Overview** — dataset stats, missing-value table, raw NO₂ time series with train/test split marker
- **Model Performance** — R², RMSE, actual vs predicted scatter, feature importance chart
- **WHO Alert System** — predicted NO₂ on test window, 200 µg/m³ threshold line, breach markers
- **Temporal Patterns** — average NO₂ by hour of day and by month

---

## Project Structure

```
air-quality-prediction/
├── air_quality_prediction.ipynb   # Full analysis notebook (all sections executed)
├── app.py                         # Streamlit dashboard (4 tabs)
├── requirements.txt               # Python dependencies
├── .gitignore
└── data/
    └── AirQualityUCI.csv          # UCI dataset (767 KB, included in repo)
```

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/diovalo/air-quality-prediction.git
cd air-quality-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook air_quality_prediction.ipynb

# 4. Or launch the dashboard
streamlit run app.py
```

---

## Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup & Data Loading | Parse semicolon/comma-decimal CSV, build datetime index |
| 2 | Exploratory Data Analysis | Completeness table, missing-value heatmap, correlation matrix |
| 3 | Data Cleaning | Replace -200 sentinels, drop missing targets, median imputation |
| 4 | Feature Engineering | Add hour_of_day, is_rush_hour, day_of_week |
| 5 | Time-Series Split | Chronological 80/20 — no shuffle, no leakage |
| 6 | Model Training | sklearn Pipeline: StandardScaler → RandomForestRegressor |
| 7 | Evaluation | MAE, RMSE, R², actual vs predicted plot, residuals histogram |
| 8 | AQI Alert Logic | Flag predictions exceeding WHO NO₂ threshold (200 µg/m³) |
| 9 | Summary | Full metrics block |
| 10 | Temporal Trend Analysis | Hourly bar chart (rush hours highlighted) + monthly line chart |

---

## Results

| Metric | Value |
|--------|-------|
| Test R² | **0.5977** |
| Test RMSE | **34.19 µg/m³** |
| Test MAE | **24.85 µg/m³** |
| Breach periods detected | **34** (predicted NO₂ > 200 µg/m³ in test window) |

**Features:** 13 total — 10 sensor readings (CO, C6H6, NOx, T, RH, AH + 4 PT08 indirect sensors) + 3 temporal features.  
**Train period:** March 2004 – January 2005 (6,172 rows)  
**Test period:** January – April 2005 (1,543 rows)

---

## Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn (Pipeline, StandardScaler, RandomForestRegressor)
- plotly (interactive charts)
- streamlit (live dashboard)

---

## Dataset

**UCI Air Quality Dataset**  
Source: https://archive.ics.uci.edu/dataset/360/air+quality

Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G. (2008).  
*On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario.*  
Sensors and Actuators B: Chemical, 129(2), 750–757.
