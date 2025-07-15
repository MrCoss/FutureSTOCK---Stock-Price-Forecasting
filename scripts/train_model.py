# scripts/train_model.py
# Author: Costas Antony Pinto

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_lstm_model(X, y, ticker, model_dir="models"):
    try:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")

        early_stop = EarlyStopping(monitor='loss', patience=5)
        model.fit(X, y, epochs=20, batch_size=32, callbacks=[early_stop], verbose=0)

        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{ticker}_model.h5")
        model.save(model_path)

        print(f"✅ Model saved: {model_path}")
        return model

    except Exception as e:
        print(f"❌ Training failed for {ticker}: {e}")
        return None


def evaluate_model(model, X, y):
    try:
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        return mse, mae, r2
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
        return None, None, None


if __name__ == "__main__":
    tickers = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META',
        'TSLA', 'NVDA', 'INTC', 'IBM',
        'JPM', 'BAC', 'WFC',
        'PFE', 'JNJ', 'MRK',
        'T', 'VZ', 'XOM', 'CVX'
    ]

    for ticker in tickers:
        try:
            print(f"\n🚀 Training model for {ticker}")
            X = np.load(f"data/processed/{ticker}_X.npy")
            y = np.load(f"data/processed/{ticker}_y.npy")

            model = train_lstm_model(X, y, ticker)
            if model:
                mse, mae, r2 = evaluate_model(model, X, y)
                if mse is not None:
                    print(f"📊 {ticker} Evaluation:")
                    print(f"  MSE: {mse:.4f}")
                    print(f"  MAE: {mae:.4f}")
                    print(f"  R2 Score: {r2:.4f}")

        except Exception as e:
            print(f"❌ Failed for {ticker}: {e}")
# scripts/train_model.py