# =====================================================================
# STEP 1: Import Required Libraries
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =====================================================================
# STEP 2: Load and Inspect Dataset
# =====================================================================
file_path = "smart_grid_dataset.csv"  # update name if needed
df = pd.read_csv(file_path)

print("Original Columns:")
print(df.columns.tolist())

# =====================================================================
# STEP 3: Clean Column Names (Fix Encoding Issues)
# =====================================================================
# Remove strange UTF-8 symbols like Ã‚ etc.
df.columns = df.columns.str.encode('ascii', 'ignore').str.decode('ascii')
df.columns = df.columns.str.strip()

print("\nCleaned Columns:")
print(df.columns.tolist())

# Drop missing values
df.dropna(inplace=True)

# =====================================================================
# STEP 4: Feature Selection
# =====================================================================
# Check final column names and adjust Temperature accordingly
features = [
    "Voltage (V)", "Current (A)", "Power Consumption (kW)",
    "Reactive Power (kVAR)", "Power Factor", "Solar Power (kW)",
    "Wind Power (kW)", "Grid Supply (kW)", "Voltage Fluctuation (%)",
    "Temperature (C)", "Humidity (%)", "Electricity Price (USD/kWh)"
]

target = "Predicted Load (kW)"

# Verify all features exist
missing = [f for f in features if f not in df.columns]
if missing:
    print("\n⚠️ Warning: Missing features detected:", missing)
    print("Adjusting automatically...\n")

# Try auto-matching similar feature names if encoding changed slightly
for f in missing:
    for col in df.columns:
        if f.split(" (")[0] in col:
            print(f"→ Auto-mapped '{f}' → '{col}'")
            features[features.index(f)] = col

# Select final features
X = df[features]
y = df[target]

# =====================================================================
# STEP 5: Scaling the Features
# =====================================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# =====================================================================
# STEP 6: Train-Test Split
# =====================================================================
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_scaled, y_scaled, test_size=0.2, shuffle=False
)

# =====================================================================
# STEP 7: Linear Regression
# =====================================================================
lr = LinearRegression()
lr.fit(X_train_base, y_train_base)
y_pred_lr = lr.predict(X_test_base)

r2_lr = r2_score(y_test_base, y_pred_lr)
mae_lr = mean_absolute_error(y_test_base, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_base, y_pred_lr))

# =====================================================================
# STEP 8: Random Forest
# =====================================================================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_base, y_train_base.ravel())
y_pred_rf = rf.predict(X_test_base)

r2_rf = r2_score(y_test_base, y_pred_rf)
mae_rf = mean_absolute_error(y_test_base, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_base, y_pred_rf))

# =====================================================================
# STEP 9 (Revised): Sort Data by Timestamp and Create Sequences
# =====================================================================
# Sort by time (VERY important for LSTM)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')

# Rebuild scaled arrays with sorted data
X = df[features]
y = df[target]
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Function for rolling sequence creation
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 24  # use 24 time steps = 6 hours if 15 min intervals
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print("Sequential dataset shape:", X_train.shape, y_train.shape)

# =====================================================================
# STEP 10 (Revised): LSTM Model
# =====================================================================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_split=0.1,
    verbose=1
)

# =====================================================================
# STEP 11 (Revised): Evaluate
# =====================================================================
y_pred_lstm = model.predict(X_test)
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred_lstm)

r2_lstm = r2_score(y_test_inv, y_pred_inv)
mae_lstm = mean_absolute_error(y_test_inv, y_pred_inv)
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

print(f"\n✅ LSTM R²: {r2_lstm:.4f} | MAE: {mae_lstm:.4f} | RMSE: {rmse_lstm:.4f}")


# =====================================================================
# STEP 12: Compare Models
# =====================================================================
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "LSTM"],
    "R2 Score": [r2_lr, r2_rf, r2_lstm],
    "MAE": [mae_lr, mae_rf, mae_lstm],
    "RMSE": [rmse_lr, rmse_rf, rmse_lstm]
})

print("\n================== MODEL PERFORMANCE COMPARISON ==================")
print(results)
print("==================================================================")

# =====================================================================
# STEP 13: Plot LSTM Training Loss
# =====================================================================
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("LSTM Training Performance")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()
