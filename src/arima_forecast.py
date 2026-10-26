import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

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

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Close"])
df = df.sort_values(by="Date").reset_index(drop=True)

data = df["Close"]

print(f"Rows loaded: {len(data)}")

# ----------------------------------
# TRAIN TEST SPLIT
# ----------------------------------

train_size = int(len(data) * 0.8)

train = data[:train_size]
test  = data[train_size:]

print(f"Train: {len(train)}  |  Test: {len(test)}")

# ----------------------------------
# TRAIN MODEL
# ----------------------------------

print("\nFitting ARIMA model (train set)...")

model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# ----------------------------------
# EVALUATION
# ----------------------------------

predictions = model_fit.forecast(steps=len(test))

mae  = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

# Safe MAPE
test_arr = test.values
nonzero  = test_arr != 0
mape = (
    np.mean(np.abs((test_arr[nonzero] - predictions.values[nonzero]) / test_arr[nonzero]))
) * 100 if nonzero.sum() > 0 else float("nan")

print("\nARIMA MODEL REPORT")
print("=" * 50)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")

# ----------------------------------
# FUTURE FORECAST (full data)
# ----------------------------------

print("\nFitting ARIMA on full data for future forecast...")

full_model = ARIMA(data, order=(5, 1, 0))
full_fit   = full_model.fit()

future_forecast = full_fit.forecast(steps=30)

print("\nNext 30-Day Forecast")
print("=" * 50)
print(future_forecast)

# ----------------------------------
# SAVE FORECAST
# ----------------------------------

forecast_df = pd.DataFrame({
    "Day":      range(1, 31),
    "Forecast": future_forecast.values
})

output_path = "reports/arima_forecast.csv"
forecast_df.to_csv(output_path, index=False)
print(f"\nForecast saved: {output_path}")

# ----------------------------------
# PLOT
# ----------------------------------

plt.figure(figsize=(14, 7))

plt.plot(
    range(len(data)),
    data.values,
    label="Historical"
)

plt.plot(
    range(len(data), len(data) + 30),
    future_forecast.values,
    label="30-Day Forecast",
    color="orange"
)

plt.title("ARIMA Stock Forecast")
plt.xlabel("Trading Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/arima_forecast_chart.png", dpi=100)
plt.close()

print("Chart saved: reports/arima_forecast_chart.png")

# ----------------------------------
# SAVE MODEL COMPARISON
# ----------------------------------

comparison_path = "reports/model_comparison.csv"

try:
    comp_df = pd.read_csv(comparison_path)
    comp_df = comp_df[comp_df["Model"] != "ARIMA"]
except FileNotFoundError:
    comp_df = pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE"])

arima_row = pd.DataFrame([{
    "Model": "ARIMA",
    "MAE":   round(mae, 2),
    "RMSE":  round(rmse, 2),
    "MAPE":  round(mape, 2)
}])

comp_df = pd.concat([comp_df, arima_row], ignore_index=True)
comp_df.to_csv(comparison_path, index=False)
print(f"Model comparison updated: {comparison_path}")

print("\nARIMA script complete.")