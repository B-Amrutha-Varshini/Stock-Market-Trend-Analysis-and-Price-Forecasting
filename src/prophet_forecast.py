import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

os.makedirs("reports", exist_ok=True)

# ----------------------------------
# LOAD DATA
# ----------------------------------

print("\nLoading dataset...")

df = pd.read_csv(
    "data/processed/reliance_processed.csv"
)

df = df[["Date", "Close"]].copy()
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df = df.dropna(subset=["ds", "y"])
df = df[df["y"] > 0].sort_values("ds").reset_index(drop=True)

print(f"Rows loaded: {len(df)}")

# ----------------------------------
# TRAIN TEST SPLIT
# ----------------------------------

train_size = int(len(df) * 0.8)

train = df[:train_size]
test  = df[train_size:]

print(f"Train: {len(train)}  |  Test: {len(test)}")

# ----------------------------------
# TRAIN MODEL ON TRAIN SET
# ----------------------------------

print("\nFitting Prophet model (train set)...")

model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True
)

model.fit(train)

# ----------------------------------
# EVALUATION
# ----------------------------------

future_test = model.make_future_dataframe(periods=len(test))
forecast_test = model.predict(future_test)

predictions = (
    forecast_test["yhat"]
    .tail(len(test))
    .reset_index(drop=True)
)

test_y = test["y"].reset_index(drop=True)

mae  = mean_absolute_error(test_y, predictions)
rmse = np.sqrt(mean_squared_error(test_y, predictions))

# Safe MAPE
nonzero = test_y.values != 0
mape = (
    np.mean(
        np.abs((test_y.values[nonzero] - predictions.values[nonzero]) / test_y.values[nonzero])
    )
) * 100 if nonzero.sum() > 0 else float("nan")

print("\nPROPHET MODEL REPORT")
print("=" * 50)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")

# ----------------------------------
# TRAIN ON FULL DATA
# ----------------------------------

print("\nFitting Prophet on full data for future forecast...")

final_model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True
)

final_model.fit(df)

future = final_model.make_future_dataframe(periods=30)
forecast = final_model.predict(future)

# ----------------------------------
# SAVE FORECAST
# ----------------------------------

output_path = "reports/prophet_forecast.csv"

forecast[
    [
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper"
    ]
].tail(30).to_csv(output_path, index=False)

print(f"\nForecast saved: {output_path}")

# ----------------------------------
# FORECAST VISUALIZATION
# ----------------------------------

fig = final_model.plot(forecast)
plt.title("Prophet Forecast")
plt.tight_layout()
plt.savefig("reports/prophet_forecast_chart.png", dpi=100)
plt.close()
print("Chart saved: reports/prophet_forecast_chart.png")

# ----------------------------------
# COMPONENTS
# ----------------------------------

fig2 = final_model.plot_components(forecast)
plt.tight_layout()
plt.savefig("reports/prophet_components.png", dpi=100)
plt.close()
print("Chart saved: reports/prophet_components.png")

# ----------------------------------
# SAVE MODEL COMPARISON
# ----------------------------------

comparison_path = "reports/model_comparison.csv"

try:
    comp_df = pd.read_csv(comparison_path)
    comp_df = comp_df[comp_df["Model"] != "Prophet"]
except FileNotFoundError:
    comp_df = pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE"])

prophet_row = pd.DataFrame([{
    "Model": "Prophet",
    "MAE":   round(mae, 2),
    "RMSE":  round(rmse, 2),
    "MAPE":  round(mape, 2)
}])

comp_df = pd.concat([comp_df, prophet_row], ignore_index=True)
comp_df.to_csv(comparison_path, index=False)
print(f"Model comparison updated: {comparison_path}")

print("\nProphet script complete.")