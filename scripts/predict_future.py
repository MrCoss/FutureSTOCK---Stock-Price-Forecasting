# scripts/predict_future.py
# Author: Costas Antony Pinto

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

def predict_next_price(ticker, processed_dir="data/processed", model_dir="models", plot_dir="plots"):
    try:
        # Paths
        X_path = os.path.join(processed_dir, f"{ticker}_X.npy")
        y_path = os.path.join(processed_dir, f"{ticker}_y.npy")
        scaler_path = os.path.join(processed_dir, f"{ticker}_scaler.pkl")
        model_path = os.path.join(model_dir, f"{ticker}_model.h5")

        # Validations
        if not all(os.path.exists(p) for p in [X_path, y_path, scaler_path, model_path]):
            raise FileNotFoundError(f"Missing file for {ticker}: check .npy, .pkl, or .h5 in paths.")

        # Load data
        X = np.load(X_path)
        y = np.load(y_path)
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)

        # Predict next price using the last available sequence
        last_sequence = X[-1].reshape(1, X.shape[1], 1)
        predicted_scaled = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

        print(f"✅ {ticker} - Predicted next close: ${predicted_price:.2f}")

        # Plot last 100 true values + predicted
        actual_prices = scaler.inverse_transform(y[-100:])
        all_actual = np.append(actual_prices, predicted_price)
        days = list(range(1, 101)) + [101]

        plt.figure(figsize=(10, 5))
        plt.plot(days[:-1], actual_prices, label="Actual Prices (last 100)")
        plt.plot(101, predicted_price, 'ro', label="Predicted Next Price")
        plt.title(f"{ticker} - Prediction")
        plt.xlabel("Time Step")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)

        # Save plot
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{ticker}_prediction.png")
        plt.savefig(plot_path)
        plt.close()

    except Exception as e:
        print(f"❌ Prediction failed for {ticker}: {e}")


# Run predictions for all tickers
if __name__ == "__main__":
    tickers = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META',
        'TSLA', 'NVDA', 'INTC', 'IBM',
        'JPM', 'BAC', 'WFC',
        'PFE', 'JNJ', 'MRK',
        'T', 'VZ', 'XOM', 'CVX'
    ]

    for ticker in tickers:
        predict_next_price(ticker)
