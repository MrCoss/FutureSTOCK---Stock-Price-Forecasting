# FutureSTOCK: A Deep Learning Stock Price Forecasting Application

[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](https://shields.io/)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Framework: TensorFlow & Streamlit](https://img.shields.io/badge/Framework-TensorFlow%20%26%20Streamlit-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Costas Antony Pinto
**Tech Stack:** Python · Streamlit · TensorFlow · Keras · NumPy · Scikit-learn · Matplotlib · Seaborn

---

## Table of Contents
- [1. Project Rationale & Business Value](#1-project-rationale--business-value)
- [2. Core Features & Functionality](#2-core-features--functionality)
- [3. The Technical Forecasting Pipeline](#3-the-technical-forecasting-pipeline)
- [4. Project Structure Explained](#4-project-structure-explained)
- [5. Setup & Usage Guide](#5-setup--usage-guide)
- [6. Production-Ready Design & Best Practices](#6-production-ready-design--best-practices)
- [7. Development Roadmap](#7-development-roadmap)
- [8. Skills Demonstrated](#8-skills-demonstrated)

---

## 1. Project Rationale & Business Value

Stock market forecasting is one of the most challenging and compelling problems in finance. Accurate predictions of future stock prices can provide significant advantages to investors, traders, and financial institutions by enabling more informed, data-driven decisions.

**FutureSTOCK** addresses this challenge by leveraging the power of **Long Short-Term Memory (LSTM)** neural networks, which are specifically designed to recognize patterns in time-series data. This project moves beyond a simple proof-of-concept to a fully modular and scalable application that can train, evaluate, and deploy individual forecasting models for a portfolio of major stocks.

**Business Value:**
- **For Investors:** Provides quantitative insights to supplement fundamental and technical analysis.
- **For Financial Analysts:** Offers a powerful tool for short-term price movement forecasting.
- **For Data Scientists:** Serves as a production-grade template for building and deploying time-series forecasting solutions.

<img width="5818" height="515" alt="Application Workflow Diagram" src="https://github.com/user-attachments/assets/30ebaf99-a89a-409f-868a-bf5689c428ff" />

---

## 2. Core Features & Functionality

This application provides a rich, interactive experience for stock price analysis and forecasting.

- **Ticker-Specific LSTM Models:** Trains and deploys a unique LSTM model for each of the 19 supported stock tickers (e.g., AAPL, GOOGL, MSFT), ensuring that predictions are based on the specific historical patterns of each company.
- **Automated Data Pipeline:** A robust set of scripts handles data fetching, preprocessing, and feature scaling (`MinMaxScaler`) automatically.
- **Comprehensive Model Evaluation:** For each model, the application calculates and displays key regression metrics, including **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R² Score**.
- **Interactive Visualizations:** The Streamlit dashboard includes dynamic plots for:
    - Historical Closing Prices
    - Actual vs. Predicted Price Comparison
    - Residuals Distribution to analyze model error
- **Real-Time Forecasting:** Users can get an instant prediction for the next trading day's closing price for any selected stock.
- **Modular and Reproducible:** The codebase is heavily modularized into distinct scripts for data preparation, training, and prediction, promoting reusability and ease of maintenance.

<img width="1000" height="400" alt="Prediction Plot" src="https://github.com/user-attachments/assets/8e59526c-71cb-45ba-a200-14b7950086a9" />

---

## 3. The Technical Forecasting Pipeline

The project is architected as a sequential pipeline, where each module performs a specific task.

1.  **Data Preparation (`scripts/prepare_data.py`):**
    - Fetches historical stock data for all 19 tickers.
    - Cleans and preprocesses the data.
    - Creates time-step sequences (e.g., using the last 60 days of data to predict the next day).
    - Applies `MinMaxScaler` to normalize the features between 0 and 1, a crucial step for neural network stability.
    - Saves the processed data and the fitted scaler object for each ticker.

2.  **Model Training (`scripts/train_models.py`):**
    - Iterates through each stock's processed data.
    - Defines and compiles an **LSTM-based neural network architecture** using TensorFlow/Keras.
    - Trains the model on the historical time-series data.
    - Evaluates the trained model on a validation set to prevent overfitting.
    - Saves the final trained model (`.h5` file) for each ticker in the `models/` directory.

3.  **Future Prediction (`scripts/predict_future.py`):**
    - Loads a specified trained model and its corresponding scaler.
    - Takes the most recent sequence of data points.
    - Uses the model to forecast the next day's closing price.
    - Re-scales the predicted value back to its original price format.

4.  **Interactive Dashboard (`app/app.py`):**
    - This is the user-facing component built with Streamlit.
    - It provides a dropdown menu to select a stock ticker.
    - It dynamically loads the relevant model, data, and plots.
    - It orchestrates calls to the prediction script to deliver real-time forecasts.
    - It presents all information—plots, metrics, and predictions—in a clean and interactive web interface.

<img width="1000" height="600" alt="Dashboard Screenshot" src="https://github.com/user-attachments/assets/127cd2f3-5c0e-419b-b0dd-e83cae2f4bd8" />

---

## 4. Project Structure Explained

The repository follows a professional, modular structure for scalability and clarity.

```

stock\_prediction\_project/
├── app/                  \# Contains all code for the Streamlit web application.
├── data/                 \# Stores raw and processed CSV data for each stock.
├── models/               \# Stores trained .h5 model files and scaler objects.
├── notebooks/            \# Jupyter notebooks for EDA, experimentation, and research.
├── plots/                \# Directory for saving static plot images generated by scripts.
├── scripts/              \# Core Python scripts for data prep, training, and prediction logic.
├── venv/                 \# Python virtual environment (excluded by .gitignore).
├── .gitignore            \# Specifies files and directories for Git to ignore.
├── README.md             \# This documentation file.
└── requirements.txt      \# A list of all Python dependencies for the project.

````

---

## 5. Setup & Usage Guide

Follow these steps to run the complete pipeline and launch the web application.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/MrCoss/FutureSTOCK.git](https://github.com/MrCoss/FutureSTOCK.git)
    cd FutureSTOCK
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute the Full Pipeline (Data Prep -> Training -> Prediction):**
    The scripts are designed to be run in sequence.
    ```bash
    # Step 1: Prepare and scale the data
    python scripts/prepare_data.py

    # Step 2: Train an LSTM model for each stock
    python scripts/train_models.py

    # Step 3 (Optional): Run a batch prediction test
    python scripts/predict_future.py
    ```

5.  **Launch the Streamlit Web Application:**
    ```bash
    streamlit run app/app.py
    ```
    Your browser will automatically open with the interactive dashboard.

---

## 6. Production-Ready Design & Best Practices

- **Modularity:** The separation of concerns into distinct scripts (`prepare_data`, `train_models`, etc.) makes the system easy to maintain, test, and upgrade.
- **Exception Handling:** All scripts include `try...except` blocks to gracefully handle potential errors like file not found or data processing issues.
- **Version Control:** A comprehensive `.gitignore` file is used to exclude unnecessary files (e.g., `__pycache__`, virtual environments, large data files) from the repository.
- **Model Isolation:** Saving individual models and scalers for each stock ticker prevents data leakage and ensures that each forecast is tailored to the specific asset.

---

## 7. Development Roadmap

- [ ] **Multi-Step Forecasting:** Enhance the model to predict prices for multiple days into the future (e.g., next 7 days).
- [ ] **Sentiment Analysis Integration:** Incorporate NLP models to analyze financial news headlines and use sentiment scores as an additional input feature.
- [ ] **Containerization:** Dockerize the application for consistent deployment across different environments.
- [ ] **Cloud Deployment:** Deploy the application on a cloud platform like Streamlit Cloud, Hugging Face Spaces, or AWS.
- [ ] **CI/CD Automation:** Implement GitHub Actions for continuous integration and automated testing.

---

## 8. Skills Demonstrated

- **Time-Series Forecasting:** Deep understanding of time-series data, stationarity, and sequence modeling.
- **Deep Learning:** Design, training, and evaluation of LSTM networks using TensorFlow and Keras.
- **Full-Stack Application Development:** Integration of a machine learning backend with a user-facing Streamlit frontend.
- **Software Engineering Best Practices:** Emphasis on clean code, modular design, reusability, version control, and exception handling.
- **Data Engineering:** Building an automated pipeline for data ingestion, preprocessing, and storage.
````
