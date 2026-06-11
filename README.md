# Air Quality Prediction

A machine learning pipeline that predicts **NO₂ (nitrogen dioxide) concentration** from real sensor data and fires alerts when predicted levels exceed the WHO safety threshold.

**Dataset:** UCI Air Quality — hourly sensor readings from Perugia, Italy (March 2004 – April 2005)  
**Model:** Random Forest Regressor | **Test R²: 0.5977** | **RMSE: 34.19 µg/m³**

---

## Overview

Nitrogen dioxide is a primary indicator of urban air pollution, produced mainly by vehicle exhaust and combustion. This project builds an end-to-end AQI forecasting pipeline:

- Loads and parses real multi-sensor time-series data
- Cleans sentinel-encoded missing values and imputes with column medians
- Engineers temporal features (hour of day, rush-hour flag, day of week)
- Trains a Random Forest model inside a scikit-learn Pipeline
- Evaluates with a chronological 80/20 split — no data leakage
- Flags hourly periods where predicted NO₂ exceeds the **200 µg/m³ WHO guideline**
- Visualises hourly and monthly NO₂ trends

---

## Project Structure

```
air-quality-prediction/
├── air_quality_prediction.ipynb   # Main notebook (all sections executed)
├── requirements.txt               # Python dependencies
├── .gitignore
└── data/                          # Not tracked — download instructions below
    └── AirQualityUCI.csv
```

---

## Dataset

**UCI Air Quality Dataset**  
Source: https://archive.ics.uci.edu/dataset/360/air+quality  
Direct download: https://archive.ics.uci.edu/static/public/360/air+quality.zip

Download the zip, extract, and place `AirQualityUCI.csv` in the `data/` folder.  
The CSV file is excluded from this repository (785 KB, semicolon-separated, European decimal format).

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/diovalo/air-quality-prediction.git
cd air-quality-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    Place AirQualityUCI.csv in data/ (see Dataset section above)

# 4. Run the notebook
jupyter notebook air_quality_prediction.ipynb
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
| Alert periods detected | **34** (NO₂ > 200 µg/m³ in test window) |

**Features used:** 13 total — 10 sensor readings (CO, C6H6, NOx, T, RH, AH + 4 PT08 indirect sensors) and 3 temporal features.  
**Train period:** March 2004 – January 2005 (6,172 rows)  
**Test period:** January – April 2005 (1,543 rows)

---

## Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn (Pipeline, StandardScaler, RandomForestRegressor)
- matplotlib, seaborn

---

## Dataset Citation

Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G. (2008).  
*On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario.*  
Sensors and Actuators B: Chemical, 129(2), 750–757.  
UCI ML Repository: https://archive.ics.uci.edu/dataset/360/air+quality
