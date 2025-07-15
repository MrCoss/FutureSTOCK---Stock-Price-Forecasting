# app.py
# Author: Costas Antony Pinto

import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
MODEL_DIR = "models"
DATA_DIR = "data/processed"

# Map ticker to company name
TICKER_MAP = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet Inc.',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corporation',
    'INTC': 'Intel Corporation',
    'IBM': 'IBM Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'BAC': 'Bank of America Corporation',
    'WFC': 'Wells Fargo & Company',
    'PFE': 'Pfizer Inc.',
    'JNJ': 'Johnson & Johnson',
    'MRK': 'Merck & Co., Inc.',
    'T': 'AT&T Inc.',
    'VZ': 'Verizon Communications Inc.',
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation'
}

st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.title("FutureSTOCK - Stock Price Forecasting")

# Sidebar - company selection
company_name = st.sidebar.selectbox("Select a Company", list(TICKER_MAP.values()))
selected_ticker = [ticker for ticker, name in TICKER_MAP.items() if name == company_name][0]
pred_days = st.sidebar.slider("Days to Predict Ahead", 1, 10, 1)

@st.cache_data
def load_data(ticker):
    X = np.load(os.path.join(DATA_DIR, f"{ticker}_X.npy"))
    y = np.load(os.path.join(DATA_DIR, f"{ticker}_y.npy"))
    scaler = joblib.load(os.path.join(DATA_DIR, f"{ticker}_scaler.pkl"))
    return X, y, scaler

@st.cache_resource
def load_model(ticker):
    return tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{ticker}_model.h5"))

def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true, label='Actual', color='blue')
    ax.plot(y_pred, label='Predicted', color='orange')
    ax.set_title("Actual vs Predicted Closing Prices")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price ($)")
    ax.legend()
    st.pyplot(fig)

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(residuals, kde=True, ax=ax, color='purple')
    ax.set_title("Residuals Distribution")
    st.pyplot(fig)

if selected_ticker:
    try:
        X, y, scaler = load_data(selected_ticker)
        model = load_model(selected_ticker)

        # Model evaluation
        y_pred = model.predict(X)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_true_inv = scaler.inverse_transform(y)

        mse = mean_squared_error(y_true_inv, y_pred_inv)
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        r2 = r2_score(y_true_inv, y_pred_inv)

        st.subheader(f"Forecast Metrics for {company_name} ({selected_ticker})")
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{mse:.4f}")
        col2.metric("MAE", f"{mae:.4f}")
        col3.metric("R² Score", f"{r2:.4f}")

        # Future prediction
        st.subheader("Next Day Forecast")
        future_input = X[-1:].reshape(1, X.shape[1], 1)
        future_pred_scaled = model.predict(future_input)
        future_pred = scaler.inverse_transform(future_pred_scaled)[0][0]
        st.success(f"Predicted next close price: **${future_pred:.2f}**")

        # Plots
        st.subheader("Prediction Plot")
        plot_predictions(y_true_inv[-100:], y_pred_inv[-100:])

        st.subheader("Residual Error Distribution")
        plot_residuals(y_true_inv.flatten(), y_pred_inv.flatten())

    except Exception as e:
        st.error(f"Failed to process {company_name}: {e}")
