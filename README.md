# 📈 FAANG Stock Price Forecasting

End-to-end time series forecasting project that predicts closing stock prices for **META, AAPL, AMZN, NFLX, and GOOGL** using classical machine learning and deep learning. The pipeline covers data collection, exploratory analysis, feature engineering, model training (Linear Regression + LSTM), and evaluation/visualization.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-f7931e?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧭 Project Overview

This project builds a complete forecasting workflow for 5 major tech stocks (FAANG: **F**acebook/Meta, **A**pple, **A**mazon, **N**etflix, **G**oogle) over the period **2019–2024**. It compares a simple statistical baseline (Linear Regression) against a deep learning sequence model (LSTM) to evaluate which approach captures stock price movement more effectively.

**Key objectives:**
- Collect and clean historical OHLCV stock data for 5 tickers
- Explore trends, correlations, returns, and volatility across stocks
- Engineer technical indicators (moving averages, RSI, volatility, lag features)
- Train and compare Linear Regression vs. LSTM models
- Visualize and evaluate forecast accuracy

---

## 🗂️ Repository Structure

```
faang-stock-forecasting/
│
├── data/
│   ├── raw/                    # Raw OHLCV data per ticker (CSV)
│   └── processed/              # Feature-engineered train/test splits
│
├── images/
│   ├── 01_closing_prices.png
│   ├── 02_normalized_growth.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_daily_returns.png
│   ├── 05_volatility.png
│   ├── 06_actual_vs_predicted.png
│   └── 07_hero_chart.png
│
├── notebooks/
│   ├── 01_data_collection.ipynb       # Fetch & store raw stock data
│   ├── 02_eda.ipynb                   # Exploratory data analysis
│   ├── 03_preprocessing.ipynb         # Feature engineering & train/test split
│   ├── 04_modeling.ipynb              # Linear Regression & LSTM training
│   └── 05_evaluation_and_viz.ipynb    # Metrics, charts & final comparison
│
├── LICENSE
└── README.md
```

---

## ⚙️ Workflow

| Step | Notebook | Description |
|------|----------|-------------|
| 1️⃣ | `01_data_collection.ipynb` | Downloads historical daily OHLCV data for META, AAPL, AMZN, NFLX, and GOOGL |
| 2️⃣ | `02_eda.ipynb` | Visualizes price trends, normalized growth, correlations, returns & volatility |
| 3️⃣ | `03_preprocessing.ipynb` | Adds technical indicators and splits each stock into time-based train/test sets |
| 4️⃣ | `04_modeling.ipynb` | Trains a Linear Regression baseline and an LSTM deep learning model per stock |
| 5️⃣ | `05_evaluation_and_viz.ipynb` | Computes RMSE/MAE and generates actual-vs-predicted comparison charts |

---

## 🧪 Feature Engineering

For every stock, the following features are engineered from the raw closing price:

| Feature | Description |
|---------|-------------|
| `MA7`, `MA21` | 7-day and 21-day moving averages |
| `Daily_Return` | Day-over-day percentage change |
| `Volatility` | 21-day rolling standard deviation of returns |
| `RSI` | 14-day Relative Strength Index (momentum indicator) |
| `Lag_1`, `Lag_2`, `Lag_3` | Closing prices from the previous 1–3 days |

Data is split **chronologically** (80% train / 20% test) — never shuffled — to prevent lookahead bias.

---

## 🤖 Models

| Model | Type | Purpose |
|-------|------|---------|
| **Linear Regression** | Statistical baseline | Fast, interpretable benchmark using engineered features |
| **LSTM** | Deep learning (TensorFlow/Keras) | Learns sequential patterns from 60-day price windows |

**LSTM architecture:** `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)`, trained with early stopping on validation loss.

---

## 📊 Results

### Closing Price Trends (2019–2024)
![Closing Prices](images/01_closing_prices.png)

### Normalized Growth Comparison (Base = 100)
![Normalized Growth](images/02_normalized_growth.png)

### Closing Price Correlation Heatmap
![Correlation Heatmap](images/03_correlation_heatmap.png)

### Daily Returns
![Daily Returns](images/04_daily_returns.png)

### 30-Day Rolling Volatility
![Volatility](images/05_volatility.png)

### Actual vs Predicted — Linear Regression vs LSTM
![Actual vs Predicted](images/06_actual_vs_predicted.png)

---

## 📈 Model Performance (Test Set)

| Stock | Linear Regression RMSE | Linear Regression MAE | LSTM RMSE | LSTM MAE |
|-------|:----------------------:|:----------------------:|:---------:|:--------:|
| META  | 9.37  | 6.34  | 114.17 | 101.14 |
| AAPL  | 2.29  | 1.69  | 12.32  | 9.97   |
| AMZN  | 2.84  | 2.16  | 20.40  | 17.58  |
| NFLX  | 0.99  | 0.73  | 3.53   | 2.88   |
| GOOGL | 2.42  | 1.66  | 12.89  | 10.60  |

**Key takeaway:** the Linear Regression baseline (trained on engineered technical features) consistently outperformed the LSTM in this setup, likely because the feature-based model directly leverages near-term lag prices, while the LSTM relies purely on raw sequential closing prices without exogenous features.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (Linear Regression), TensorFlow/Keras (LSTM)
- **Environment:** Jupyter Notebook

---

## 🚀 Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/aditya-datahub/faang-stock-forecasting.git
   cd faang-stock-forecasting
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
   ```
3. Run the notebooks in order (01 → 05) from the `notebooks/` folder.

---

## 🔮 Future Improvements

- Add exogenous features (news sentiment, macroeconomic indicators) to the LSTM
- Experiment with GRU, Transformer, or hybrid ARIMA-LSTM architectures
- Hyperparameter tuning via grid/Bayesian search
- Deploy as an interactive forecasting dashboard (Streamlit/Dash)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Aditya** — [GitHub](https://github.com/aditya-datahub)

If you found this project useful, consider ⭐ starring the repo!
