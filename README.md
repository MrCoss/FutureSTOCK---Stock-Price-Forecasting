# FutureSTOCK - Stock Price Forecasting with LSTM

**Author:** Costas Antony Pinto  
**Tech Stack:** Python · Streamlit · TensorFlow · NumPy · Scikit-learn · Matplotlib · Seaborn  

---

## 🚀 Project Overview

**FutureSTOCK** is a robust, modular, and production-grade stock price forecasting application. It uses LSTM-based deep learning models to predict future closing prices of top publicly traded companies (AAPL, GOOGL, MSFT, etc.).

Built with scalability and professional-grade modularization in mind, the project includes:

- Secure, exception-handled data preprocessing
- Stock-specific LSTM model training and evaluation
- Future prediction for next-day closing prices
- Fully interactive **Streamlit web app** with plots, metrics, and real-time forecasting
<img width="5818" height="515" alt="Blank diagram" src="https://github.com/user-attachments/assets/30ebaf99-a89a-409f-868a-bf5689c428ff" />
<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/8e59526c-71cb-45ba-a200-14b7950086a9" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/127cd2f3-5c0e-419b-b0dd-e83cae2f4bd8" />

---

## 🧠 Key Features

✅ LSTM-based model for time-series forecasting  
✅ Automated preprocessing & scaling (MinMaxScaler)  
✅ Individual model training per stock (19 total tickers)  
✅ Metrics: MSE, MAE, R² Score  
✅ Visualizations: Actual vs Predicted, Residuals  
✅ Modular Python scripts for reproducibility  
✅ Clean Git structure and `.gitignore`  
✅ One-click deployment ready with Streamlit  

---

## 🗂️ Project Structure

```

stock\_prediction\_project/
├── app/                 # Streamlit app logic
├── data/                # Raw & processed stock data
├── models/              # Trained .h5 model files
├── notebooks/           # EDA, experiments, R\&D
├── plots/               # Saved plot images
├── scripts/             # Data prep, training, prediction scripts
├── venv/                # Virtual environment (excluded in git)
├── .gitignore
├── README.md
└── requirements.txt

````

---

## 🛠️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/stock-forecasting-app.git
cd stock-forecasting-app
````

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Data

```bash
python scripts/prepare_data.py
```

### 5. Train Models

```bash
python scripts/train_models.py
```

### 6. Predict Future Prices

```bash
python scripts/predict_future.py
```

### 7. Launch Streamlit App

```bash
streamlit run app/app.py
```

---

## 📊 Visuals (Sample)

* **Prediction vs Actual Prices**
* **Residual Distribution**
* **Interactive Stock Selector (Streamlit Sidebar)**
* **Forecasted Next Day Close Price**

---

## 🔐 Security & Best Practices

* All scripts include structured exception handling
* `.gitignore` excludes sensitive/binary files
* Models and scalers saved per ticker (safe modular isolation)
* TensorFlow retracing warnings handled via optimized design

---

## 📌 Future Enhancements

* Incorporate sentiment analysis from news headlines
* Add multi-step (multi-day) forecasting
* Dockerize the application
* Deploy on Hugging Face Spaces / AWS / Streamlit Cloud
* Add CI/CD pipelines and testing

---

## 📚 Relevant Skills Demonstrated

* Time Series Forecasting
* Deep Learning (LSTM)
* Model Evaluation and Deployment
* Python Programming and OOP
* Real-time Data Applications
* Clean Code, Reusability, and Modularization

---

## 🧾 License

Feel free to use, modify, or contribute.

---

> Built with purpose and precision by **Costas Antony Pinto**.
