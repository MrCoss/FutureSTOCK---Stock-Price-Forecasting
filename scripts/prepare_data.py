# scripts/prepare_data.py
# Author: Costas Antony Pinto

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_and_prepare_data(ticker, input_dir="data/raw", output_dir="data/processed", window_size=60):
    os.makedirs(output_dir, exist_ok=True)

    try:
        input_path = os.path.join(input_dir, f"{ticker}_data.csv")
        df = pd.read_csv(input_path, skiprows=2)  # skip first 2 rows (headers and ticker row)
        df['Date'] = pd.read_csv(input_path, skiprows=1, nrows=1).columns[0]  # manually insert 'Date' column header
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)

        if 'Close' not in df.columns:
            raise ValueError(f"No 'Close' column in {ticker} data.")

        # Normalize closing price
        close_prices = df[['Close']].values
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(close_prices)

        # Save scaler for inverse_transform during prediction
        scaler_path = os.path.join(output_dir, f"{ticker}_scaler.pkl")
        joblib.dump(scaler, scaler_path)

        # Create sequences
        X, y = [], []
        for i in range(window_size, len(scaled_prices)):
            X.append(scaled_prices[i - window_size:i])
            y.append(scaled_prices[i])

        X = np.array(X)
        y = np.array(y)

        np.save(os.path.join(output_dir, f"{ticker}_X.npy"), X)
        np.save(os.path.join(output_dir, f"{ticker}_y.npy"), y)

        print(f"✅ {ticker} data processed and saved: X={X.shape}, y={y.shape}")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")


# Batch process all tickers
if __name__ == "__main__":
    tickers = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META',
        'TSLA', 'NVDA', 'INTC', 'IBM',
        'JPM', 'BAC', 'WFC',
        'PFE', 'JNJ', 'MRK',
        'T', 'VZ', 'XOM', 'CVX'
    ]

    for ticker in tickers:
        load_and_prepare_data(ticker)
