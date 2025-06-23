import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from io import BytesIO

st.set_page_config(page_title="WFM Forecasting Tool", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1e40af;
        cursor: pointer;
    }
    .stFileUploader, .stSelectbox, .stNumberInput, .stSlider, .stDateInput {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
    }
    .stExpander {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: #64748b;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb;
        border-bottom: 2px solid #2563eb;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .metric-table {
        background-color: #eff6ff;
        padding: 10px;
        border-radius: 8px;
    }
    .success-message {
        background-color: #dcfce7;
        color: #166534;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #2563eb;'>WFM Forecasting Tool</h2>", unsafe_allow_html=True)
    st.markdown("A powerful tool for workforce planning with advanced forecasting and interval analysis.")
    theme = st.selectbox("Theme", ["Light", "Dark"], key="theme_select")
    if theme == "Dark":
        st.markdown("""
            <style>
            .main { background-color: #1e293b; color: #f1f5f9; }
            .stExpander, .stFileUploader, .stSelectbox, .stNumberInput, .stSlider, .stDateInput {
                background-color: #334155;
                border-color: #475569;
                color: #f1f5f9;
            }
            h1, h2, h3 { color: #f1f5f9; }
            .metric-table { background-color: #334155; }
            .success-message { background-color: #166534; color: #dcfce7; }
            </style>
        """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Version**: 1.0 | **Date**: June 20, 2025")

# Initialize session state
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None
if "monthly_forecast" not in st.session_state:
    st.session_state.monthly_forecast = None
if "results" not in st.session_state:
    st.session_state.results = None
if "error_measures" not in st.session_state:
    st.session_state.error_measures = None

def generate_download_link(df, filename, content_type="text/csv"):
    """Generate a download link for a DataFrame as CSV or Excel."""
    try:
        if content_type == "csv":
            output = df.to_csv(index=False)
            b64 = base64.b64encode(output.encode()).decode()
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/{content_type};base64,{b64}" download="{filename}">Download {filename}</a>'
        return href
    except Exception as e:
        st.error(f"Error generating download link: {e}")
        return ""

def generate_forecast_template():
    """Generate a sample template for forecasting data."""
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    channels = ["Chat", "Email", "Phone"]
    data = {
        "Date": dates.repeat(len(channels)),
        "Channel": np.tile(channels, len(dates)),
        "Volume": np.random.randint(100, 1000, size=len(dates) * len(channels))
    }
    return pd.DataFrame(data)

def generate_interval_template():
    """Generate a sample template for interval data."""
    dates = pd.date_range(start="2024-01-01", end="2024-01-07", freq="D")
    intervals = [f"{h:02d}:00" for h in range(24)]
    data = {"Date": dates.repeat(len(intervals)), "Interval": np.tile(intervals, len(dates))}
    df = pd.DataFrame(data)
    for interval in intervals:
        df[interval] = np.random.randint(10, 100, size=len(df))
    return df

# Define tabs
tab1, tab2 = st.tabs(["📈 Forecast App", "⏰ Interval Forecast"])

### TAB 1: Forecasting Tool ###
with tab1:
    st.header("Workforce Forecasting")
    st.markdown("Generate accurate volume forecasts using advanced models and calculate FTE requirements.")

    # Template download
    with st.expander("📥 Download Sample Template", expanded=False):
        st.markdown("Download a sample CSV to understand the required format (`Date`, `Channel`, `Volume`).")
        forecast_template = generate_forecast_template()
        st.markdown(generate_download_link(forecast_template, "forecast_template.csv", content_type="csv"), unsafe_allow_html=True)

    # File upload
    with st.expander("📤 Upload Data", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx"],
            key="forecast_upload",
            help="File must contain 'Date', 'Channel', and 'Volume' columns."
        )
        data_dict = {}
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file, parse_dates=["Date"], dayfirst=True)
                    data_dict["Sheet1"] = df
                else:
                    data_dict = pd.read_excel(uploaded_file, sheet_name=None, engine="openpyxl")
                for sheet, df in data_dict.items():
                    df.columns = df.columns.str.lower().str.strip()
                    required_cols = ["date", "channel", "volume"]
                    if not all(col in df.columns for col in required_cols):
                        st.error(f"Sheet '{sheet}' must contain 'date', 'channel', and 'volume' columns.")
                        st.stop()
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    if df["date"].isna().any():
                        st.error(f"Sheet '{sheet}' contains invalid dates. Please check the data format.")
                        st.stop()
                    if df["volume"].apply(lambda x: not isinstance(x, (int, float)) or pd.isna(x)).any():
                        st.error(f"Sheet '{sheet}' contains invalid numeric 'volume' values.")
                        st.stop()
                st.markdown("<div class='success-message'>✅ Data uploaded successfully!</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing uploaded data: {e}")
                st.stop()

    if data_dict:
        with st.expander("🔍 Select Data", expanded=True):
            selected_sheet = st.selectbox(
                "Select Sheet",
                list(data_dict.keys()),
                key="sheet_select",
                help="Choose the sheet containing your data."
            )
            data = data_dict[selected_sheet]
            st.markdown("**Dataset Preview:**")
            st.dataframe(data.head(), use_container_width=True)

            selected_channel = st.selectbox(
                "Select Channel",
                data["channel"].unique(),
                key="channel_select",
                help="Filter data by channel for forecasting."
            )

        # Scenario adjustment
        with st.expander("⚙️ Scenario Configuration", expanded=False):
            st.markdown("Adjust forecasts for campaigns, holidays, or custom scenarios.")
            scenario_type = st.selectbox(
                "Scenario Type",
                ["None", "Campaign Impact", "Holiday Impact", "Custom Scenario"],
                key="scenario_select",
                help="Choose a scenario to adjust forecast volumes."
            )
            scenario_dict = {}

            if scenario_type == "Campaign Impact":
                st.markdown("**Campaign Settings**")
                col1, col2 = st.columns(2)
                with col1:
                    uplift = st.number_input(
                        "Volume Uplift %",
                        min_value=-100.0,
                        max_value=500.0,
                        value=20.0,
                        key="uplift",
                        help="Percentage increase in volume due to campaign."
                    )
                with col2:
                    campaign_start = st.date_input("Start Date", key="campaign_start")
                campaign_end = st.date_input("End Date", key="campaign_end")
                if campaign_start > campaign_end:
                    st.error("Start date must be before end date.")
                    st.stop()
                scenario_dict = {
                    "type": "campaign",
                    "start": pd.to_datetime(campaign_start),
                    "end": pd.to_datetime(campaign_end),
                    "uplift_pct": uplift / 100
                }
            elif scenario_type == "Holiday Impact":
                holiday_file = st.file_uploader(
                    "Upload Holiday File (CSV or Excel)",
                    type=["csv", "xlsx"],
                    key="holiday_upload",
                    help="File must contain 'Date' and 'Impact %' columns."
                )
                if holiday_file:
                    try:
                        holidays_df = pd.read_csv(holiday_file) if holiday_file.name.endswith(".csv") else pd.read_excel(holiday_file, engine="openpyxl")
                        holidays_df.columns = holidays_df.columns.str.lower().str.strip()
                        if "date" not in holidays_df.columns or "impact %" not in holidays_df.columns:
                            st.error("Holiday file must contain 'date' and 'impact %' columns.")
                            st.stop()
                        holidays_df["date"] = pd.to_datetime(holidays_df["date"], errors="coerce")
                        if holidays_df["date"].isna().any():
                            st.error("Holiday file contains invalid dates.")
                            st.stop()
                        if holidays_df["impact %"].apply(lambda x: not isinstance(x, (int, float)) or pd.isna(x)).any():
                            st.error("Holiday file contains invalid 'impact %' values.")
                            st.stop()
                        st.dataframe(holidays_df, use_container_width=True)
                        scenario_dict = {"type": "holiday", "holidays_df": holidays_df}
                    except Exception as e:
                        st.error(f"Error processing holiday file: {e}")
                        st.stop()
            elif scenario_type == "Custom Scenario":
                custom_dates = st.date_input(
                    "Select Dates",
                    value=None,
                    help="Choose dates for custom volume adjustments.",
                    key="custom_dates"
                )
                adj_factor = st.number_input(
                    "Adjustment Factor",
                    value=1.2,
                    key="adj_factor",
                    help="E.g., 1.2 for a 20% increase."
                )
                if custom_dates is None or not custom_dates:
                    st.warning("No dates selected. Scenario will be ignored.")
                    scenario_dict = {}
                else:
                    custom_dates = [custom_dates] if isinstance(custom_dates, pd.Timestamp) else list(custom_dates)
                    scenario_dict = {"type": "custom", "dates": pd.to_datetime(custom_dates), "factor": adj_factor}

        def apply_scenario_adjustment(forecast_df, scenario_dict):
            """Apply scenario-based adjustments to forecast data."""
            if not scenario_dict:
                return forecast_df
            forecast_df = forecast_df.copy()
            try:
                if scenario_dict["type"] == "campaign":
                    mask = (forecast_df["Date"] >= scenario_dict["start"]) & (forecast_df["Date"] <= scenario_dict["end"])
                    for model in ["Prophet", "Holt-Winters", "ARIMA", "Ensemble"]:
                        if model in forecast_df.columns:
                            forecast_df.loc[mask, model] *= (1 + scenario_dict["uplift_pct"])
                elif scenario_dict["type"] == "holiday":
                    holiday_df = scenario_dict["holidays_df"]
                    holiday_map = dict(zip(holiday_df["date"], holiday_df["impact %"] / 100))
                    for model in ["Prophet", "Holt-Winters", "ARIMA", "Ensemble"]:
                        if model in forecast_df.columns:
                            forecast_df[model] = forecast_df.apply(
                                lambda row: row[model] * (1 + holiday_map.get(row["Date"], 0)), axis=1)
                elif scenario_dict["type"] == "custom":
                    dates = scenario_dict["dates"]
                    if isinstance(dates, pd.Timestamp):
                        dates = [dates]
                    mask = forecast_df["Date"].isin(dates)
                    for model in ["Prophet", "Holt-Winters", "ARIMA", "Ensemble"]:
                        if model in forecast_df.columns:
                            forecast_df.loc[mask, model] *= scenario_dict["factor"]
                return forecast_df
            except Exception as e:
                st.error(f"Error applying scenario adjustment: {e}")
                return forecast_df

        def forecast_with_methods(df, channel):
            """Generate forecasts using Prophet, Holt-Winters, ARIMA, and Ensemble."""
            df = df[df["channel"] == channel].copy()
            if df.empty or len(df) < 37:
                st.error("Insufficient data. Ensure at least 37 data points for the channel.")
                return None, None, None, None
            df.set_index("date", inplace=True)
            df = df.asfreq("D").fillna(method="ffill")
            validation_period = 30
            train_df = df.iloc[:-validation_period]
            valid_df = df.iloc[-validation_period:]
            actuals = valid_df["volume"]
            forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=180, freq="D")
            forecast_data = pd.DataFrame({"Date": forecast_dates})
            error_measures = {}
            results = {}

            # Prophet
            try:
                prophet_data = train_df.reset_index().rename(columns={"date": "ds", "volume": "y"})
                prophet_model = Prophet()
                prophet_model.fit(prophet_data)
                future = prophet_model.make_future_dataframe(periods=180 + validation_period, freq="D")
                prophet_forecast = prophet_model.predict(future)
                prophet_valid = prophet_forecast.iloc[-validation_period - 180:-180]["yhat"]
                if len(prophet_valid) != validation_period:
                    st.warning(f"Prophet validation data length mismatch: {len(prophet_valid)} vs {validation_period}")
                    prophet_valid = np.nan
                forecast_data["Prophet"] = prophet_forecast.iloc[-180:]["yhat"].values.round(2)
                results["Prophet"] = prophet_forecast.iloc[-180:].set_index("ds")["yhat"]
                if not np.isnan(prophet_valid).all():
                    error_measures["Prophet"] = {
                        "MAE": round(mean_absolute_error(actuals, prophet_valid), 2),
                        "MAPE%": round(mean_absolute_percentage_error(actuals, prophet_valid) * 100, 2)
                    }
                else:
                    error_measures["Prophet"] = {"MAE": np.nan, "MAPE%": np.nan}
            except Exception as e:
                st.warning(f"Prophet failed: {e}")
                forecast_data["Prophet"] = np.nan
                error_measures["Prophet"] = {"MAE": np.nan, "MAPE%": np.nan}
                prophet_valid = np.nan

            # Holt-Winters
            try:
                seasonal_periods = 7 if len(train_df) >= 7 else None
                if seasonal_periods:
                    hw_model = ExponentialSmoothing(train_df["volume"], seasonal="add", seasonal_periods=seasonal_periods).fit()
                    hw_valid = hw_model.forecast(validation_period)
                    hw_valid = pd.Series(hw_valid, index=valid_df.index)[:validation_period]
                    if len(hw_valid) != validation_period:
                        st.warning(f"Holt-Winters validation data length mismatch: {len(hw_valid)} vs {validation_period}")
                        hw_valid = np.nan
                    hw_forecast = hw_model.forecast(validation_period + 180)[-180:]
                    forecast_data["Holt-Winters"] = hw_forecast.values.round(2)
                    results["Holt-Winters"] = pd.Series(hw_forecast.values, index=forecast_dates)
                    if not np.isnan(hw_valid).all():
                        error_measures["Holt-Winters"] = {
                            "MAE": round(mean_absolute_error(actuals, hw_valid), 2),
                            "MAPE%": round(mean_absolute_percentage_error(actuals, hw_valid) * 100, 2)
                        }
                    else:
                        error_measures["Holt-Winters"] = {"MAE": np.nan, "MAPE%": np.nan}
                else:
                    st.warning("Holt-Winters skipped: insufficient data for seasonality.")
                    forecast_data["Holt-Winters"] = np.nan
                    error_measures["Holt-Winters"] = {"MAE": np.nan, "MAPE%": np.nan}
                    hw_valid = np.nan
            except Exception as e:
                st.warning(f"Holt-Winters failed: {e}")
                forecast_data["Holt-Winters"] = np.nan
                error_measures["Holt-Winters"] = {"MAE": np.nan, "MAPE%": np.nan}
                hw_valid = np.nan

            # ARIMA
            try:
                arima_model = ARIMA(train_df["volume"], order=(1, 1, 0)).fit()
                arima_valid = arima_model.forecast(steps=validation_period)
                arima_valid = pd.Series(arima_valid, index=valid_df.index)[:validation_period]
                if len(arima_valid) != validation_period:
                    st.warning(f"ARIMA validation data length mismatch: {len(arima_valid)} vs {validation_period}")
                    arima_valid = np.nan
                arima_forecast = arima_model.forecast(steps=validation_period + 180)[-180:]
                forecast_data["ARIMA"] = arima_forecast.values.round(2)
                results["ARIMA"] = pd.Series(arima_forecast.values, index=forecast_dates)
                if not np.isnan(arima_valid).all():
                    error_measures["ARIMA"] = {
                        "MAE": round(mean_absolute_error(actuals, arima_valid), 2),
                        "MAPE%": round(mean_absolute_percentage_error(actuals, arima_valid) * 100, 2)
                    }
                else:
                    error_measures["ARIMA"] = {"MAE": np.nan, "MAPE%": np.nan}
            except Exception as e:
                st.warning(f"ARIMA failed: {e}")
                forecast_data["ARIMA"] = np.nan
                error_measures["ARIMA"] = {"MAE": np.nan, "MAPE%": np.nan}
                arima_valid = np.nan

            # Ensemble Forecast
            try:
                weights = {}
                total_inverse_mae = 0
                for model in ["Prophet", "Holt-Winters", "ARIMA"]:
                    mae = error_measures.get(model, {}).get("MAE", np.nan)
                    if not np.isnan(mae) and mae > 0:
                        weights[model] = 1 / mae
                        total_inverse_mae += weights[model]
                    else:
                        weights[model] = 0
                if total_inverse_mae > 0:
                    for model in weights:
                        weights[model] /= total_inverse_mae
                    forecast_data["Ensemble"] = (
                        weights.get("Prophet", 0) * forecast_data["Prophet"].fillna(0) +
                        weights.get("Holt-Winters", 0) * forecast_data["Holt-Winters"].fillna(0) +
                        weights.get("ARIMA", 0) * forecast_data["ARIMA"].fillna(0)
                    ).round(2)
                    ensemble_valid = np.zeros(validation_period)
                    if not np.isnan(prophet_valid).all():
                        ensemble_valid += weights.get("Prophet", 0) * prophet_valid.fillna(0).values
                    if not np.isnan(hw_valid).all():
                        ensemble_valid += weights.get("Holt-Winters", 0) * hw_valid.fillna(0).values
                    if not np.isnan(arima_valid).all():
                        ensemble_valid += weights.get("ARIMA", 0) * arima_valid.fillna(0).values
                    if len(ensemble_valid) != len(actuals):
                        st.warning(f"Ensemble validation data length mismatch: {len(ensemble_valid)} vs {len(actuals)}")
                        raise ValueError("Ensemble validation length mismatch")
                    results["Ensemble"] = pd.Series(forecast_data["Ensemble"].values, index=forecast_dates)
                    error_measures["Ensemble"] = {
                        "MAE": round(mean_absolute_error(actuals, ensemble_valid), 2),
                        "MAPE%": round(mean_absolute_percentage_error(actuals, ensemble_valid) * 100, 2)
                    }
                else:
                    st.warning("Ensemble forecast skipped: no valid model MAEs available.")
                    forecast_data["Ensemble"] = np.nan
                    error_measures["Ensemble"] = {"MAE": np.nan, "MAPE%": np.nan}
            except Exception as e:
                st.warning(f"Ensemble forecast failed: {e}")
                forecast_data["Ensemble"] = np.nan
                error_measures["Ensemble"] = {"MAE": np.nan, "MAPE%": np.nan}

            forecast_data = apply_scenario_adjustment(forecast_data, scenario_dict)
            forecast_data["Month"] = forecast_data["Date"].dt.to_period("M")
            monthly_forecast = forecast_data.groupby("Month").sum(numeric_only=True).reset_index()
            return forecast_data, monthly_forecast, error_measures, results

        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("🚀 Generate Forecast", key="forecast_button", use_container_width=True):
            with st.spinner("Generating forecasts..."):
                forecast_data, monthly_forecast, error_measures, results = forecast_with_methods(data, selected_channel)
                if forecast_data is not None and monthly_forecast is not None and results is not None:
                    st.session_state.forecast_data = forecast_data
                    st.session_state.monthly_forecast = monthly_forecast
                    st.session_state.results = results
                    st.session_state.error_measures = error_measures
                    st.markdown("<div class='success-message'>✅ Forecasts generated successfully!</div>", unsafe_allow_html=True)
                else:
                    st.error("Forecast generation failed. Please check your data.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Forecast results
        if st.session_state.forecast_data is not None:
            with st.expander("📊 Forecast Results", expanded=True):
                st.markdown("**Raw Forecast Data:**")
                st.dataframe(st.session_state.forecast_data.head(), use_container_width=True)
                st.markdown("**Monthly Aggregated Forecast:**")
                st.dataframe(st.session_state.monthly_forecast, use_container_width=True)

                st.markdown("### Error Metrics (MAE & MAPE%)")
                if st.session_state.error_measures:
                    error_df = pd.DataFrame.from_dict(st.session_state.error_measures, orient="index").reset_index()
                    error_df.columns = ["Model", "MAE", "MAPE%"]
                    st.markdown("<div class='metric-table'>", unsafe_allow_html=True)
                    st.dataframe(error_df.style.format({"MAE": "{:.2f}", "MAPE%": "{:.2f}"}).highlight_min(subset=["MAE", "MAPE%"], color="#bbf7d0"), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No error metrics available.")

                if st.session_state.results:
                    for method, values in st.session_state.results.items():
                        st.markdown(f"### {method} Forecast")
                        st.line_chart(values, use_container_width=True)
                else:
                    st.warning("No forecast results available to plot.")

                st.markdown("### 📥 Download Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(generate_download_link(st.session_state.forecast_data, "forecast_data.csv", content_type="csv"), unsafe_allow_html=True)
                with col2:
                    st.markdown(generate_download_link(st.session_state.monthly_forecast, "monthly_aggregated_forecast.csv", content_type="csv"), unsafe_allow_html=True)

        # FTE Estimation
        with st.expander("🧑‍💼 FTE Estimation", expanded=False):
            if st.session_state.forecast_data is not None and st.session_state.monthly_forecast is not None:
                st.markdown("**FTE Parameters**")
                col1, col2 = st.columns(2)
                with col1:
                    aht = st.number_input(
                        "AHT (Minutes)",
                        min_value=0.1,
                        value=4.0,
                        format="%.2f",
                        key="aht_input",
                        help="Average Handle Time per interaction."
                    )
                    occupancy = st.slider(
                        "Occupancy Rate (%)",
                        min_value=1,
                        max_value=100,
                        value=70,
                        key="occupancy_input",
                        help="Percentage of time agents are handling interactions."
                    )
                with col2:
                    production_hours = st.number_input(
                        "Production Hours per FTE",
                        min_value=1.0,
                        max_value=168.0,
                        value=135.5,
                        format="%.2f",
                        key="prod_hours_input",
                        help="Monthly productive hours per full-time employee."
                    )
                    bench_buffer = st.number_input(
                        "Bench & Buffer (%)",
                        min_value=0.0,
                        value=6.0,
                        format="%.2f",
                        key="bench_buffer_input",
                        help="Additional capacity for flexibility."
                    )
                if st.button("Calculate FTE", key="fte_calc_button"):
                    monthly_forecast = st.session_state.monthly_forecast.copy()
                    for model in ["Prophet", "Holt-Winters", "ARIMA", "Ensemble"]:
                        if model in monthly_forecast.columns:
                            monthly_forecast[f"FTE_{model}"] = (
                                ((monthly_forecast[model] * (aht / 60)) / (occupancy / 100)) / production_hours
                            ) * (1 + bench_buffer / 100)
                    st.markdown("**FTE Requirements:**")
                    st.dataframe(monthly_forecast.round(2), use_container_width=True)
                    st.session_state.monthly_forecast = monthly_forecast
                    st.markdown(generate_download_link(monthly_forecast, "fte_requirements.csv", content_type="csv"), unsafe_allow_html=True)
            else:
                st.info("Generate a forecast first to calculate FTE requirements.")

### TAB 2: Interval Forecast ###
with tab2:
    st.header("Interval Forecasting")
    st.markdown("Distribute daily forecasts into hourly intervals based on historical patterns.")

    # Template download
    with st.expander("📥 Download Sample Template", expanded=False):
        st.markdown("Download a sample Excel file to understand the interval data format.")
        interval_template = generate_interval_template()
        st.markdown(generate_download_link(interval_template, "interval_template.xlsx", content_type="vnd.openxmlformats-officedocument.spreadsheetml.sheet"), unsafe_allow_html=True)

    # File upload
    with st.expander("📤 Upload Interval Data", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            interval_file = st.file_uploader(
                "Historical Interval Data (Excel)",
                type=["xlsx", "xls"],
                key="interval_upload",
                help="Excel file with 'Date', 'Interval', and hourly columns."
            )
        with col2:
            forecast_file = st.file_uploader(
                "Daily Forecast File (Excel)",
                type=["xlsx", "xls"],
                key="forecast_file_upload",
                help="Excel file from Tab 1 with 'Date' and 'Forecast' columns."
            )

        if interval_file and forecast_file:
            try:
                interval_xl = pd.ExcelFile(interval_file)
                forecast_xl = pd.ExcelFile(forecast_file)
                all_sheets = list(set(interval_xl.sheet_names) & set(forecast_xl.sheet_names))
                if not all_sheets:
                    st.error("No common sheets found between interval and forecast files.")
                    st.stop()
                selected_sheets = st.multiselect(
                    "Select Sheets",
                    options=all_sheets,
                    default=all_sheets,
                    key="interval_sheets",
                    help="Choose sheets to process."
                )
                if selected_sheets:
                    output_dict = {}
                    for sheet in selected_sheets:
                        with st.expander(f"Processing: {sheet}", expanded=True):
                            try:
                                interval_df = pd.read_excel(interval_file, sheet_name=sheet, engine="openpyxl")
                                forecast_df = pd.read_excel(forecast_file, sheet_name=sheet, engine="openpyxl")
                                interval_df.columns = interval_df.columns.str.strip()
                                forecast_df.columns = forecast_df.columns.str.strip()
                                if 'Date' not in interval_df.columns:
                                    st.error(f"Column 'Date' not found in interval data of sheet '{sheet}'.")
                                    continue
                                if 'Date' not in forecast_df.columns or 'Forecast' not in forecast_df.columns:
                                    st.error(f"Columns 'Date' and 'Forecast' not found in forecast data of sheet '{sheet}'.")
                                    continue
                                interval_df['Date'] = pd.to_datetime(interval_df['Date'], errors='coerce')
                                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], errors='coerce')
                                if interval_df['Date'].isnull().any():
                                    st.warning(f"Some dates in interval data of sheet '{sheet}' could not be parsed.")
                                    continue
                                if forecast_df['Date'].isnull().any():
                                    st.warning(f"Some dates in forecast data of sheet '{sheet}' could not be parsed.")
                                    continue
                                forecast_df['Forecast'] = pd.to_numeric(forecast_df['Forecast'], errors='coerce').fillna(0)
                                interval_cols = [col for col in interval_df.columns if col not in ["Date", "Interval"]]
                                interval_df[interval_cols] = interval_df[interval_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                                interval_totals = interval_df[interval_cols].sum(axis=1)
                                valid_rows = interval_totals > 0
                                interval_share = interval_df[interval_cols].loc[valid_rows].div(interval_totals[valid_rows], axis=0)
                                avg_distribution = interval_share.mean().fillna(0)
                                if avg_distribution.sum() > 0:
                                    avg_distribution = avg_distribution / avg_distribution.sum()
                                else:
                                    st.error(f"Invalid interval distribution for sheet '{sheet}'. All values are zero.")
                                    continue
                                interval_forecast = pd.DataFrame()
                                for date in forecast_df['Date']:
                                    forecast_value = forecast_df.loc[forecast_df['Date'] == date, 'Forecast'].values[0]
                                    interval_forecast_for_date = pd.DataFrame({
                                        'Date': [date] * len(avg_distribution),
                                        'Interval': avg_distribution.index,
                                        'Forecast': forecast_value * avg_distribution.values
                                    })
                                    interval_forecast = pd.concat([interval_forecast, interval_forecast_for_date])
                                interval_forecast.reset_index(drop=True, inplace=True)
                                output_dict[sheet] = interval_forecast
                                st.markdown("<div class='success-message'>✅ Interval forecast completed!</div>", unsafe_allow_html=True)
                                st.dataframe(interval_forecast.head(), use_container_width=True)
                            except Exception as e:
                                st.error(f"Error processing sheet '{sheet}': {e}")
                    if output_dict:
                        st.markdown("### 📥 Download Interval Forecasts")
                        output_excel = BytesIO()
                        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                            for sheet_name, df in output_dict.items():
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                        output_excel.seek(0)
                        st.download_button(
                            "Download All Interval Forecasts",
                            data=output_excel.getvalue(),
                            file_name="interval_forecasts.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="interval_download",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error processing files: {e}")