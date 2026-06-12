import calendar

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Air Quality Prediction", layout="wide")

SENTINEL = -200
TARGET = "NO2(GT)"
NO2_ALERT_THRESHOLD = 200
FEATURES = [
    "CO(GT)", "C6H6(GT)", "NOx(GT)", "T", "RH", "AH",
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S5(O3)",
]
RUSH_HOURS = set(range(7, 10)) | set(range(16, 20))
ALL_FEATURES = FEATURES + ["hour_of_day", "is_rush_hour", "day_of_week"]


@st.cache_data
def load_data():
    df_raw = pd.read_csv(
        "data/AirQualityUCI.csv",
        sep=";",
        decimal=",",
        parse_dates=False,
    )
    df_raw = df_raw.loc[:, ~df_raw.columns.str.startswith("Unnamed")]
    df_raw["datetime"] = pd.to_datetime(
        df_raw["Date"] + " " + df_raw["Time"],
        format="%d/%m/%Y %H.%M.%S",
    )
    df_raw = df_raw.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()

    df = df_raw.copy()
    df.replace(SENTINEL, np.nan, inplace=True)

    before_pct = (df.notna().sum() / len(df) * 100).round(1)

    df_clean = df[[TARGET] + FEATURES].dropna(subset=[TARGET]).copy()
    for col in FEATURES:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    df_clean["hour_of_day"] = df_clean.index.hour
    df_clean["is_rush_hour"] = df_clean["hour_of_day"].isin(RUSH_HOURS).astype(int)
    df_clean["day_of_week"] = df_clean.index.dayofweek

    return df, before_pct, df_clean


@st.cache_resource
def get_model():
    _, _, df_clean = load_data()
    split_idx = int(len(df_clean) * 0.80)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(train_df[ALL_FEATURES], train_df[TARGET])
    y_pred = pipeline.predict(test_df[ALL_FEATURES])

    return pipeline, train_df, test_df, test_df[TARGET], y_pred


df, before_pct, df_clean = load_data()
pipeline, train_df, test_df, y_test, y_pred = get_model()

r2 = r2_score(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
split_boundary = train_df.index.max()

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Model Performance", "WHO Alert System", "Temporal Patterns",
])

with tab1:
    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Readings", "7,700+")
    c2.metric("Date Range", "Mar 2004 – Apr 2005")
    c3.metric("Location", "Perugia, Italy")
    c4.metric("Features", "13")

    st.subheader("Missing Values: Before vs After Cleaning")
    display_cols = [TARGET] + FEATURES
    mv_table = pd.DataFrame({
        "Column": display_cols,
        "Before Cleaning (%)": [float(before_pct.get(c, 0.0)) for c in display_cols],
        "After Cleaning (%)": [100.0] * len(display_cols),
    })
    st.dataframe(mv_table, hide_index=True, use_container_width=True)

    st.subheader("Raw NO₂ Over Time")
    no2_ts = df[TARGET].dropna().reset_index()
    no2_ts.columns = ["datetime", "NO2"]
    fig_ts = px.line(
        no2_ts, x="datetime", y="NO2",
        title="Raw NO₂ (NO2(GT)) Concentration Over Time",
        labels={"NO2": "NO₂ (µg/m³)", "datetime": "Date"},
    )
    fig_ts.add_vline(
        x=str(split_boundary),
        line_dash="dash",
        line_color="red",
        annotation_text="Train/Test Split",
        annotation_position="top right",
    )
    st.plotly_chart(fig_ts, use_container_width=True)


with tab2:
    st.header("Model Performance")

    c1, c2 = st.columns(2)
    c1.metric("R²", f"{r2:.4f}")
    c2.metric("RMSE", f"{rmse:.2f} µg/m³")

    scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    lo = float(min(scatter_df["Actual"].min(), scatter_df["Predicted"].min()))
    hi = float(max(scatter_df["Actual"].max(), scatter_df["Predicted"].max()))

    fig_scatter = px.scatter(
        scatter_df, x="Actual", y="Predicted",
        title="Actual vs Predicted NO₂ (Test Set)",
        labels={
            "Actual": "Actual NO₂ (µg/m³)",
            "Predicted": "Predicted NO₂ (µg/m³)",
        },
        opacity=0.5,
    )
    fig_scatter.add_shape(
        type="line", x0=lo, y0=lo, x1=hi, y1=hi,
        line=dict(color="red", dash="dash"),
    )
    fig_scatter.add_annotation(
        text="Time-series-aware split — no data leakage",
        xref="paper", yref="paper", x=0.02, y=0.98,
        showarrow=False, bgcolor="white", bordercolor="gray", borderwidth=1,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    rf_model = pipeline.named_steps["model"]
    imp_df = (
        pd.DataFrame({"Feature": ALL_FEATURES, "Importance": rf_model.feature_importances_})
        .sort_values("Importance", ascending=True)
    )
    fig_imp = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        title="Feature Importance (RandomForestRegressor)",
        labels={"Importance": "Importance Score", "Feature": ""},
    )
    st.plotly_chart(fig_imp, use_container_width=True)


with tab3:
    st.header("WHO Alert System")
    st.info(f"WHO NO₂ alert threshold: **{NO2_ALERT_THRESHOLD} µg/m³**")

    results_df = test_df[[TARGET]].copy()
    results_df["predicted_NO2"] = y_pred
    results_df["alert"] = (results_df["predicted_NO2"] > NO2_ALERT_THRESHOLD).astype(int)
    alert_count = int(results_df["alert"].sum())

    st.metric("Breach Periods Detected", f"{alert_count} high-pollution breach periods detected")

    alert_df = results_df.reset_index()
    alerts = alert_df[alert_df["alert"] == 1]

    fig_alert = px.line(
        alert_df, x="datetime", y="predicted_NO2",
        title="Predicted NO₂ on Test Window with WHO Alert Threshold",
        labels={"predicted_NO2": "Predicted NO₂ (µg/m³)", "datetime": "Date"},
    )
    fig_alert.add_hline(
        y=NO2_ALERT_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text=f"WHO Threshold ({NO2_ALERT_THRESHOLD} µg/m³)",
        annotation_position="top right",
    )
    fig_alert.add_scatter(
        x=alerts["datetime"],
        y=alerts["predicted_NO2"],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Alert Period",
    )
    st.plotly_chart(fig_alert, use_container_width=True)
    st.error(f"{alert_count} high-pollution breach periods detected")


with tab4:
    st.header("Temporal Patterns")

    c1, c2 = st.columns(2)

    with c1:
        hourly = df_clean.groupby("hour_of_day")[TARGET].mean().reset_index()
        hourly.columns = ["Hour", "Mean NO₂"]
        fig_h = px.bar(
            hourly, x="Hour", y="Mean NO₂",
            title="Average NO₂ by Hour of Day",
            labels={"Hour": "Hour of Day", "Mean NO₂": "Mean NO₂ (µg/m³)"},
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with c2:
        monthly = (
            df_clean.assign(Month=df_clean.index.month)
            .groupby("Month")[TARGET]
            .mean()
            .reset_index()
        )
        monthly.columns = ["Month", "Mean NO₂"]
        month_order = [calendar.month_abbr[m] for m in monthly["Month"]]
        monthly["Month Name"] = month_order

        fig_m = px.bar(
            monthly, x="Month Name", y="Mean NO₂",
            title="Average NO₂ by Month",
            labels={"Month Name": "Month", "Mean NO₂": "Mean NO₂ (µg/m³)"},
            category_orders={"Month Name": month_order},
        )
        st.plotly_chart(fig_m, use_container_width=True)
