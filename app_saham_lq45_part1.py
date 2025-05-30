import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========== Fungsi bantu ==========

def load_data(symbol, start_date, end_date):
    path = os.path.join("data", f"{symbol}.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return df

def create_glm_features(data, lags=5):
    df_feat = pd.DataFrame()
    for i in range(1, lags + 1):
        df_feat[f"lag_{i}"] = data.shift(i)
    df_feat["target"] = data.values
    return df_feat.dropna()

def evaluate(true, pred):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mse, rmse, mape

# ========== Sidebar ==========

st.sidebar.title("Data Download")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., BBCA)", value="BBCA")
start_date = st.sidebar.date_input("Start Date", datetime(2019, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2023, 1, 1))
model_type = st.sidebar.selectbox("Select Model Type", ["GLM", "ARIMA"])

# ========== Main Title ==========

st.title("Stock Price Prediction Web App")

# ========== Load & Process ==========

try:
    df = load_data(symbol, start_date, end_date)
    close_prices = df['close'].dropna().reset_index(drop=True)

    if model_type == "GLM":
        features = create_glm_features(close_prices)
        X = sm.add_constant(features.drop(columns="target"))
        y = features["target"]
        model = sm.GLM(y, X).fit()
        pred = model.predict(X)
        mse, rmse, mape = evaluate(y, pred)

    elif model_type == "ARIMA":
        stepwise_model = auto_arima(close_prices, start_p=1, start_q=1,
                                     max_p=3, max_q=3, seasonal=False,
                                     d=None, trace=False,
                                     error_action='ignore',
                                     suppress_warnings=True, stepwise=True)
        best_order = stepwise_model.order
        model = ARIMA(close_prices, order=best_order).fit()
        pred = model.predict(start=best_order[1], end=len(close_prices)-1)
        true = close_prices[best_order[1]:]
        mse, rmse, mape = evaluate(true, pred)

    # ========== Results ==========

    st.subheader(f"Results for {model_type} Model")
    st.markdown(f"**Mean Squared Error (MSE):** `{mse:.2f}`")
    st.markdown(f"**Root Mean Squared Error (RMSE):** `{rmse:.2f}`")
    st.markdown(f"**Mean Absolute Percentage Error (MAPE):** `{mape:.2f}` %")

    # ========== Visualization ==========

    st.subheader("Visualize Predictions")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(close_prices.index, close_prices, label="Actual Stock Prices", color="blue")
    if model_type == "GLM":
        ax.plot(features.index, pred, label="Predicted Stock Prices", color="red")
    else:
        ax.plot(range(best_order[1], len(close_prices)), pred, label="Predicted Stock Prices", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price (IDR)")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
