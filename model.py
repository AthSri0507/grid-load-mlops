# =============================================================================
# SMART GRID LOAD PREDICTION - MODEL BUILDING WITH 3 ALGORITHMS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# STEP 1: LOAD DATASET
# =============================================================================
file_path = "smart_grid_dataset (1).csv"   # change if filename differs
df = pd.read_csv(file_path)

# Fix column encoding issues and trim names
df.columns = [col.encode('latin1').decode('utf-8').strip() for col in df.columns]

# Drop timestamp (not numeric)
df.drop(columns=['Timestamp'], inplace=True, errors='ignore')

# Drop rows with missing values
df.dropna(inplace=True)

# =============================================================================
# STEP 2: FEATURE SELECTION
# =============================================================================
target = 'Predicted Load (kW)'
if target not in df.columns:
    raise ValueError(f"❌ Column '{target}' not found in dataset!")

X = df.drop(columns=[target])
y = df[target]

# Convert any categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# Normalize numeric data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# =============================================================================
# STEP 3: RANDOM FOREST REGRESSOR
# =============================================================================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\n🌲 RANDOM FOREST RESULTS")
print(f"R² Score: {rf_r2:.4f}")
print(f"MAE: {rf_mae:.4f}")
print(f"RMSE: {rf_rmse:.4f}")

# =============================================================================
# STEP 4: LINEAR REGRESSION
# =============================================================================
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

print("\n📈 LINEAR REGRESSION RESULTS")
print(f"R² Score: {lr_r2:.4f}")
print(f"MAE: {lr_mae:.4f}")
print(f"RMSE: {lr_rmse:.4f}")

# =============================================================================
# STEP 5: LSTM MODEL
# =============================================================================
# Reshape for LSTM (samples, timesteps, features)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y, test_size=0.2, random_state=42
)

# Define model
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mse')

print("\n🧠 Training LSTM model (this may take a moment)...")
history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

lstm_pred = lstm_model.predict(X_test_lstm).flatten()

lstm_r2 = r2_score(y_test_lstm, lstm_pred)
lstm_mae = mean_absolute_error(y_test_lstm, lstm_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_pred))

print("\n⚡ LSTM RESULTS")
print(f"R² Score: {lstm_r2:.4f}")
print(f"MAE: {lstm_mae:.4f}")
print(f"RMSE: {lstm_rmse:.4f}")

# =============================================================================
# STEP 6: COMPARISON PLOT
# =============================================================================
models = ['Random Forest', 'Linear Regression', 'LSTM']
r2_scores = [rf_r2, lr_r2, lstm_r2]

plt.figure(figsize=(7, 5))
plt.bar(models, r2_scores, color=['green', 'blue', 'orange'])
plt.ylabel('R² Score')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.show()
