# scripts/fetch_data.py
# Author: Costas Antony Pinto

import os
import yfinance as yf
import pandas as pd

def download_stock_data(tickers, start_date, end_date, save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)

    for ticker in tickers:
        try:
            print(f"🔽 Downloading data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)

            if data is None or data.empty:
                print(f"⚠️ No data found for {ticker}. Skipping.")
                continue

            file_path = os.path.join(save_dir, f"{ticker}_data.csv")
            data.to_csv(file_path)
            print(f"✅ {ticker} data saved at {file_path}\n")

        except Exception as e:
            print(f"❌ Failed to fetch data for {ticker}: {e}")

# 🔁 Run manually when script is executed standalone
if __name__ == "__main__":
    tickers = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META',     # Tech
        'TSLA', 'NVDA', 'INTC', 'IBM',               # Hardware/AI
        'JPM', 'BAC', 'WFC',                         # Banks
        'PFE', 'JNJ', 'MRK',                         # Pharma
        'T', 'VZ',                                   # Telecom
        'XOM', 'CVX'                                 # Oil & Energy
    ]

    # ⏳ Set the timeframe
    start_date = "2015-01-01"
    end_date = "2024-12-31"

    # 🚀 Trigger download
    download_stock_data(tickers, start_date, end_date)
