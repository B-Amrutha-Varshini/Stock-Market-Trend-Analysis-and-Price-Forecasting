
import streamlit as st
import pandas as pd

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Market Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FB;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        color: #17365D;
        letter-spacing: -0.5px;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }

    .section-title {
        color: #17365D;
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }

    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        min-height: 145px;
    }

    .info-card h4 {
        color: #17365D;
        margin-top: 0;
        font-size: 1.05rem;
    }

    .info-card p {
        color: #64748B;
        font-size: 0.92rem;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 5px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }

    div[data-testid="stMetricValue"] {
        color: #17365D;
    }

    [data-testid="stSidebar"] {
        background-color: #EDF2F8;
    }

    .status-note {
        background-color: #EAF2FF;
        border-left: 4px solid #3478F6;
        padding: 12px 16px;
        border-radius: 5px;
        color: #17365D;
        margin: 10px 0 18px 0;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        padding-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. MODEL EVALUATION DATA
# ============================================================
# These are the example MAPE values supplied for the project.
# Replace them with the actual evaluation results when available.

MODEL_DATA = pd.DataFrame({
    "Model": ["Prophet", "ARIMA", "LSTM"],
    "MAPE (%)": [17.08, 9.48, 6.73]
})

best_model = MODEL_DATA.loc[
    MODEL_DATA["MAPE (%)"].idxmin(), "Model"
]

best_mape = MODEL_DATA["MAPE (%)"].min()


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 📈 Market Intelligence")
    st.caption("STOCK ANALYTICS PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Exploratory Analysis",
            "Technical Analysis",
            "Risk Analysis",
            "Forecasting Models",
            "About Project"
        ],
        index=0
    )

    st.divider()

    st.markdown("### Platform Modules")

    st.markdown("""
    - Historical market analysis
    - Technical indicators
    - Risk evaluation
    - Time-series forecasting
    - Model performance comparison
    """)

    st.divider()
    st.caption("Stock Market Intelligence Platform")
    st.caption("Analytics Dashboard | Version 1.0")


# ============================================================
# 5. MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Stock Market Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Forecasting, Technical Analysis and Risk Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 6. OVERVIEW DASHBOARD
# ============================================================

if page == "Overview":

    st.markdown("### Executive Overview")

    st.write(
        "A centralized analytical dashboard for examining historical "
        "market behaviour, comparing forecasting models, and exploring "
        "technical indicators and financial risk."
    )

    # KPI CARDS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Forecasting Models",
            value=f"{len(MODEL_DATA):02d}"
        )

    with col2:
        st.metric(
            label="Best Performing Model",
            value=best_model
        )

    with col3:
        st.metric(
            label="Lowest MAPE",
            value=f"{best_mape:.2f}%"
        )

    with col4:
        st.metric(
            label="Models Evaluated",
            value=str(len(MODEL_DATA))
        )

    # MODEL PERFORMANCE
    st.markdown(
        '<div class="section-title">Forecasting Performance</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1.4])

    with left:
        st.markdown("#### Model Evaluation Summary")

        st.dataframe(
            MODEL_DATA.sort_values("MAPE (%)"),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn(
                    "Forecasting Model"
                ),
                "MAPE (%)": st.column_config.NumberColumn(
                    "MAPE (%)",
                    format="%.2f%%"
                )
            }
        )

        st.markdown(
            f"""
            <div class="status-note">
                <strong>Best result:</strong> {best_model} has the
                lowest MAPE among the displayed evaluation results.
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown("#### Mean Absolute Percentage Error")

        chart_data = MODEL_DATA.set_index("Model")
        st.bar_chart(
            chart_data[["MAPE (%)"]],
            color="#3478F6",
            horizontal=True
        )

        st.caption(
            "Lower MAPE indicates lower average percentage error "
            "on the evaluated data. It does not guarantee better "
            "future performance."
        )

    # ANALYTICS MODULES
    st.markdown(
        '<div class="section-title">Analytics Modules</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Exploratory Analysis</h4>
            <p>Investigate historical prices, market trends,
            return distributions and data characteristics.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h4>📉 Technical Analysis</h4>
            <p>Examine indicators such as moving averages,
            RSI and MACD when available in the dataset.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h4>🛡️ Risk Analytics</h4>
            <p>Assess volatility, historical drawdowns,
            returns and other risk measurements.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Project Highlights</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    - **Data processing:** Prepare and validate historical market data.
    - **Feature engineering:** Create features for analysis and prediction.
    - **Exploratory analysis:** Identify patterns in price and return data.
    - **Risk analysis:** Evaluate historical risk measures.
    - **Forecasting:** Compare ARIMA, Prophet and LSTM model results.
    """)


# ============================================================
# 7. EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "Exploratory Analysis":

    st.subheader("Exploratory Data Analysis")

    st.write(
        "Explore historical stock prices and understand the structure "
        "and behaviour of the market dataset."
    )

    uploaded_file = st.file_uploader(
        "Upload historical stock data (CSV)",
        type=["csv"],
        key="eda_upload"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.markdown("#### Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)

            st.markdown("#### Dataset Information")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Total Records", f"{len(df):,}")

            with m2:
                st.metric("Total Columns", str(len(df.columns)))

            with m3:
                st.metric(
                    "Missing Values",
                    f"{int(df.isna().sum().sum()):,}"
                )

            st.markdown("#### Summary Statistics")
            st.dataframe(
                df.describe(include="all").T,
                use_container_width=True
            )

            date_candidates = [
                col for col in df.columns
                if col.lower() in ["date", "datetime", "timestamp"]
            ]

            price_candidates = [
                col for col in df.columns
                if col.lower() in [
                    "close", "adj close", "adj_close",
                    "closing_price", "price"
                ]
            ]

            if date_candidates and price_candidates:
                date_col = date_candidates[0]
                price_col = price_candidates[0]

                plot_df = df[[date_col, price_col]].copy()
                plot_df[date_col] = pd.to_datetime(
                    plot_df[date_col], errors="coerce"
                )
                plot_df[price_col] = pd.to_numeric(
                    plot_df[price_col], errors="coerce"
                )

                plot_df = plot_df.dropna().sort_values(date_col)

                if not plot_df.empty:
                    st.markdown("#### Historical Price Trend")

                    st.line_chart(
                        plot_df.set_index(date_col)[price_col],
                        y_label="Price"
                    )
            else:
                st.info(
                    "For a price trend chart, include a date column "
                    "and a closing-price column in your CSV."
                )

        except Exception as e:
            st.error(f"Unable to read the CSV file: {e}")

    else:
        st.info(
            "Upload a CSV file containing historical stock market data "
            "to explore the dataset and display its price trend."
        )


# ============================================================
# 8. TECHNICAL ANALYSIS
# ============================================================

elif page == "Technical Analysis":

    st.subheader("Technical Analysis")

    st.write(
        "Review commonly used technical indicators for evaluating "
        "historical price momentum and trends."
    )

    st.markdown("#### Supported Indicator Concepts")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown("""
        <div class="info-card">
            <h4>Moving Average</h4>
            <p>Helps smooth short-term price fluctuations and
            identify potential trend direction.</p>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class="info-card">
            <h4>Relative Strength Index</h4>
            <p>RSI is a momentum indicator commonly displayed
            on a scale from 0 to 100.</p>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class="info-card">
            <h4>MACD</h4>
            <p>Compares exponential moving averages to help
            evaluate changes in momentum.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Calculate Indicators from CSV")

    technical_file = st.file_uploader(
        "Upload CSV with Date and Close columns",
        type=["csv"],
        key="technical_upload"
    )

    if technical_file is not None:
        try:
            tech_df = pd.read_csv(technical_file)

            close_col = next(
                (
                    col for col in tech_df.columns
                    if col.lower() in ["close", "adj close", "adj_close"]
                ),
                None
            )

            date_col = next(
                (
                    col for col in tech_df.columns
                    if col.lower() in ["date", "datetime", "timestamp"]
                ),
                None
            )

            if close_col is None:
                st.error(
                    "A Close or Adj Close column is required."
                )
            else:
                tech_df[close_col] = pd.to_numeric(
                    tech_df[close_col], errors="coerce"
                )

                tech_df = tech_df.dropna(subset=[close_col]).copy()

                if date_col:
                    tech_df[date_col] = pd.to_datetime(
                        tech_df[date_col], errors="coerce"
                    )
                    tech_df = tech_df.dropna(subset=[date_col])
                    tech_df = tech_df.sort_values(date_col)

                tech_df["SMA 20"] = (
                    tech_df[close_col].rolling(20).mean()
                )

                tech_df["SMA 50"] = (
                    tech_df[close_col].rolling(50).mean()
                )

                daily_change = tech_df[close_col].diff()

                gains = daily_change.clip(lower=0)
                losses = -daily_change.clip(upper=0)

                avg_gain = gains.ewm(
                    alpha=1 / 14, min_periods=14, adjust=False
                ).mean()

                avg_loss = losses.ewm(
                    alpha=1 / 14, min_periods=14, adjust=False
                ).mean()

                relative_strength = avg_gain / avg_loss

                tech_df["RSI 14"] = (
                    100 - (100 / (1 + relative_strength))
                )

                ema12 = tech_df[close_col].ewm(
                    span=12, adjust=False
                ).mean()

                ema26 = tech_df[close_col].ewm(
                    span=26, adjust=False
                ).mean()

                tech_df["MACD"] = ema12 - ema26

                tech_df["Signal Line"] = tech_df["MACD"].ewm(
                    span=9, adjust=False
                ).mean()

                st.markdown("#### Price and Moving Averages")

                chart_cols = [close_col, "SMA 20", "SMA 50"]

                if date_col:
                    price_chart = tech_df.set_index(date_col)[chart_cols]
                else:
                    price_chart = tech_df[chart_cols]

                st.line_chart(price_chart)

                st.markdown("#### RSI")
                rsi_chart = (
                    tech_df.set_index(date_col)["RSI 14"]
                    if date_col else tech_df["RSI 14"]
                )
                st.line_chart(rsi_chart)

                st.markdown("#### MACD")
                macd_chart = (
                    tech_df.set_index(date_col)[["MACD", "Signal Line"]]
                    if date_col
                    else tech_df[["MACD", "Signal Line"]]
                )
                st.line_chart(macd_chart)

                st.markdown("#### Indicator Data")
                st.dataframe(
                    tech_df.tail(30),
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Unable to calculate indicators: {e}")

    else:
        st.info(
            "Upload historical stock prices to calculate moving averages, "
            "RSI and MACD."
        )


# ============================================================
# 9. RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.subheader("Risk Analysis")

    st.write(
        "Calculate basic historical risk measures from a CSV containing "
        "daily closing prices."
    )

    risk_file = st.file_uploader(
        "Upload CSV with a Close column",
        type=["csv"],
        key="risk_upload"
    )

    if risk_file is not None:
        try:
            risk_df = pd.read_csv(risk_file)

            close_col = next(
                (
                    col for col in risk_df.columns
                    if col.lower() in ["close", "adj close", "adj_close"]
                ),
                None
            )

            date_col = next(
                (
                    col for col in risk_df.columns
                    if col.lower() in ["date", "datetime", "timestamp"]
                ),
                None
            )

            if close_col is None:
                st.error(
                    "The uploaded file must contain a Close or Adj Close column."
                )
            else:
                risk_df[close_col] = pd.to_numeric(
                    risk_df[close_col], errors="coerce"
                )

                risk_df = risk_df.dropna(subset=[close_col]).copy()

                if date_col:
                    risk_df[date_col] = pd.to_datetime(
                        risk_df[date_col], errors="coerce"
                    )
                    risk_df = risk_df.dropna(subset=[date_col])
                    risk_df = risk_df.sort_values(date_col)

                prices = risk_df[close_col]
                returns = prices.pct_change().dropna()

                if len(returns) < 2:
                    st.warning(
                        "At least three valid closing-price observations "
                        "are needed for these risk calculations."
                    )
                else:
                    annualized_volatility = (
                        returns.std(ddof=1) * (252 ** 0.5) * 100
                    )

                    cumulative_returns = (1 + returns).cumprod()

                    running_peak = cumulative_returns.cummax()

                    drawdown = (
                        cumulative_returns / running_peak
                    ) - 1

                    maximum_drawdown = drawdown.min() * 100

                    mean_daily_return = returns.mean() * 100

                    r1, r2, r3 = st.columns(3)

                    with r1:
                        st.metric(
                            "Annualized Volatility",
                            f"{annualized_volatility:.2f}%"
                        )

                    with r2:
                        st.metric(
                            "Maximum Drawdown",
                            f"{maximum_drawdown:.2f}%"
                        )

                    with r3:
                        st.metric(
                            "Average Daily Return",
                            f"{mean_daily_return:.3f}%"
                        )

                    st.markdown("#### Daily Returns")

                    return_chart = pd.DataFrame({
                        "Daily Return (%)": returns * 100
                    })

                    if date_col:
                        return_chart.index = risk_df[date_col].iloc[
                            1:
                        ].values

                    st.line_chart(return_chart)

                    st.markdown("#### Historical Drawdown")

                    drawdown_chart = pd.DataFrame({
                        "Drawdown (%)": drawdown * 100
                    })

                    if date_col:
                        drawdown_chart.index = risk_df[date_col].iloc[
                            1:
                        ].values

                    st.area_chart(drawdown_chart)

                    st.caption(
                        "Volatility is annualized using 252 trading days. "
                        "Historical measures are not guarantees of future risk."
                    )

        except Exception as e:
            st.error(f"Unable to calculate risk measures: {e}")

    else:
        st.info(
            "Upload a CSV with historical closing prices to calculate "
            "volatility, drawdown and daily returns."
        )


# ============================================================
# 10. FORECASTING MODELS
# ============================================================

elif page == "Forecasting Models":

    st.subheader("Forecasting Model Comparison")

    st.write(
        "Compare the example evaluation results for Prophet, ARIMA and LSTM."
    )

    st.markdown("#### Evaluation Metric")

    metric = st.selectbox(
        "Select metric",
        ["MAPE (%)"]
    )

    st.dataframe(
        MODEL_DATA.sort_values("MAPE (%)"),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Model Performance Chart")

    st.bar_chart(
        MODEL_DATA.set_index("Model")[[metric]],
        horizontal=True,
        color="#3478F6"
    )

    st.markdown("#### Interpretation")

    st.write(
        f"Among the supplied results, {best_model} has the lowest "
        f"MAPE at {best_mape:.2f}%. The Prophet result is 17.08%, "
        "the ARIMA result is 9.48%, and the LSTM result is 6.73%."
    )

    st.warning(
        "These are the example values supplied for this dashboard. "
        "They are not automatically loaded from your model scripts. "
        "Replace them with measured results from the same test dataset "
        "and evaluation period before drawing conclusions."
    )

    st.markdown("#### Upload Actual Forecast Results")

    results_file = st.file_uploader(
        "Upload model results as CSV",
        type=["csv"],
        key="forecast_upload"
    )

    if results_file is not None:
        try:
            results_df = pd.read_csv(results_file)

            st.dataframe(
                results_df,
                use_container_width=True
            )

            st.caption(
                "The uploaded results are displayed for inspection. "
                "The example comparison above remains unchanged."
            )

        except Exception as e:
            st.error(f"Unable to read forecast results: {e}")


# ============================================================
# 11. ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.subheader("About the Project")

    st.write(
        "The Stock Market Intelligence Platform combines data preparation, "
        "exploratory analysis, technical indicators, risk measurements "
        "and time-series forecasting in a single application."
    )

    st.markdown("#### Project Objectives")

    st.markdown("""
    1. Collect and prepare historical stock market data.
    2. Validate and clean market records.
    3. Engineer useful features for analysis and forecasting.
    4. Explore price movements and historical market behaviour.
    5. Evaluate financial risk using historical returns.
    6. Compare ARIMA, Prophet and LSTM forecasting approaches.
    7. Present analytical results through an interactive dashboard.
    """)

    st.markdown("#### Technology Stack")

    stack_df = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "Streamlit",
            "ARIMA",
            "Prophet",
            "TensorFlow / Keras"
        ],
        "Purpose": [
            "Application logic and data analysis",
            "Data preparation and manipulation",
            "Interactive dashboard",
            "Time-series forecasting",
            "Forecasting and trend modelling",
            "LSTM neural network forecasting"
        ]
    })

    st.dataframe(
        stack_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Important Disclaimer")

    st.info(
        "This application is intended for educational and analytical purposes. "
        "Historical performance and model forecasts do not guarantee future "
        "stock prices or investment returns."
    )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Stock Market Intelligence Platform<br>
        Historical Analysis | Technical Indicators | Risk Analytics | Forecasting
    </div>
    """,
    unsafe_allow_html=True
)
