# scripts/plot_predictions.py
# Author: Costas Pinto
# Secure and robust script to generate prediction plots

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

def plot_predictions(ticker, processed_dir="data/processed", models_dir="models", plots_dir="plots"):
    try:
        os.makedirs(plots_dir, exist_ok=True)

        # Load the last 100 timesteps from processed data
        X = np.load(os.path.join(processed_dir, f"{ticker}_X.npy"))
        y = np.load(os.path.join(processed_dir, f"{ticker}_y.npy"))
        scaler = joblib.load(os.path.join(processed_dir, f"{ticker}_scaler.pkl"))
        model = load_model(os.path.join(models_dir, f"{ticker}_model.h5"))

        # Pick last 100 examples for plotting
        X_plot = X[-100:]
        y_true = y[-100:]

        # Predict
        y_pred = model.predict(X_plot)

        # Inverse transform
        y_true_rescaled = scaler.inverse_transform(y_true)
        y_pred_rescaled = scaler.inverse_transform(y_pred)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(y_true_rescaled, label="Actual", color="blue")
        plt.plot(y_pred_rescaled, label="Predicted", color="orange")
        plt.title(f"{ticker} - Actual vs Predicted Closing Prices (Last 100 Days)")
        plt.xlabel("Days")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_path = os.path.join(plots_dir, f"{ticker}_prediction.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"✅ Plot saved for {ticker}: {plot_path}")

    except Exception as e:
        print(f"❌ Plotting failed for {ticker}: {e}")

if __name__ == "__main__":
    tickers = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META',
        'TSLA', 'NVDA', 'INTC', 'IBM',
        'JPM', 'BAC', 'WFC',
        'PFE', 'JNJ', 'MRK',
        'T', 'VZ', 'XOM', 'CVX'
    ]

    for ticker in tickers:
        plot_predictions(ticker)
