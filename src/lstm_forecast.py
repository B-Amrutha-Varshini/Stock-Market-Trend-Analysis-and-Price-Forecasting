import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
except (ImportError, ModuleNotFoundError):
    try:
        from keras.models import Sequential
        from keras.layers import Dense, LSTM
    except (ImportError, ModuleNotFoundError):
        print(
            "\n[ERROR] Neither 'tensorflow' nor 'keras' could be imported.\n"
            "  Fix: Run the following in your terminal and retry:\n"
            "       pip install tensorflow --upgrade\n"
            "  If you get a long-path error, enable Windows Long Path support:\n"
            "       Run as Administrator: reg add HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f\n"
            "       Then restart and retry.\n"
        )
        raise SystemExit(1)

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

os.makedirs("reports", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading dataset...")

df = pd.read_csv(
    "data/processed/reliance_processed.csv"
)

# =====================================================
# DATE SORT & VALIDATION
# =====================================================

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values(by="Date").reset_index(drop=True)

# =====================================================
# VALIDATE CLOSE COLUMN
# =====================================================

if "Close" not in df.columns:
    raise ValueError("ERROR: 'Close' column not found in dataset.")

df = df.dropna(subset=["Close"])
df = df[df["Close"] > 0].reset_index(drop=True)

print(f"Dataset loaded: {len(df)} rows, from {df['Date'].min().date()} to {df['Date'].max().date()}")

data = df["Close"].values.reshape(-1, 1)

# =====================================================
# TRAIN/TEST SPLIT BEFORE SCALING (avoid data leakage)
# =====================================================

sequence_length = 60
train_size = int(len(data) * 0.8)

train_data = data[:train_size]
test_data  = data[train_size:]

# =====================================================
# SCALING (fit ONLY on training data)
# =====================================================

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)

# Scale full data using train-fitted scaler
scaled_all = scaler.transform(data)

# =====================================================
# CREATE SEQUENCES
# =====================================================

def create_sequences(data_array, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data_array)):
        X.append(data_array[i - seq_len:i, 0])
        y.append(data_array[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(scaled_all[:train_size + sequence_length], sequence_length)
X_test,  y_test  = create_sequences(scaled_all[train_size:], sequence_length)

# Reshape for LSTM: (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# =====================================================
# BUILD MODEL
# =====================================================

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.summary()

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining LSTM model...")

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    verbose=1
)

# =====================================================
# PREDICTIONS ON TEST SET
# =====================================================

predictions_scaled = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(predictions_scaled)
actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

# =====================================================
# EVALUATION
# =====================================================

mae  = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

# Safe MAPE: avoid division by zero
nonzero_mask = actual.flatten() != 0
if nonzero_mask.sum() > 0:
    mape = (
        np.mean(
            np.abs(
                (actual[nonzero_mask] - predictions[nonzero_mask])
                / actual[nonzero_mask]
            )
        )
    ) * 100
else:
    mape = float("nan")

print("\nLSTM EVALUATION")
print("=" * 60)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")

# =====================================================
# FUTURE FORECAST (next 30 days)
# =====================================================

future_input = scaled_all[-sequence_length:].reshape(1, sequence_length, 1)
future_predictions = []

current_batch = future_input.copy()

for _ in range(30):
    next_pred = model.predict(current_batch, verbose=0)
    future_predictions.append(next_pred[0][0])
    current_batch = np.concatenate(
        (current_batch[:, 1:, :], next_pred.reshape(1, 1, 1)),
        axis=1
    )

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# =====================================================
# SAVE FORECAST CSV
# =====================================================

forecast_df = pd.DataFrame({
    "Day": range(1, 31),
    "Forecast": future_predictions.flatten()
})

output_path = "reports/lstm_forecast.csv"
forecast_df.to_csv(output_path, index=False)

print(f"\nForecast saved to: {output_path}")
print("\nNext 30-Day Price Forecast:")
print(forecast_df.to_string(index=False))

# =====================================================
# SAVE MODEL COMPARISON REPORT
# =====================================================

comparison_path = "reports/model_comparison.csv"

# Read existing comparison if available, else create fresh
try:
    comp_df = pd.read_csv(comparison_path)
    # Update or insert LSTM row
    comp_df = comp_df[comp_df["Model"] != "LSTM"]
except FileNotFoundError:
    comp_df = pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE"])

lstm_row = pd.DataFrame([{
    "Model": "LSTM",
    "MAE":   round(mae, 2),
    "RMSE":  round(rmse, 2),
    "MAPE":  round(mape, 2)
}])

comp_df = pd.concat([comp_df, lstm_row], ignore_index=True)
comp_df.to_csv(comparison_path, index=False)

print(f"\nModel comparison updated: {comparison_path}")
print("\nDone.")