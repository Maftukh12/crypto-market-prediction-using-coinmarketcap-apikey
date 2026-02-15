# Crypto ML Prediction System

Sistem machine learning untuk menganalisis dan memprediksi arah market cryptocurrency menggunakan data dari CoinMarketCap API.

## Features

- **Data Collection**: Mengambil data real-time dari CoinMarketCap API
- **Technical Analysis**: 20+ technical indicators (RSI, MACD, Bollinger Bands, dll)
- **Machine Learning Models**:
  - LSTM Neural Network untuk time-series prediction
  - Random Forest Classifier untuk market direction
  - XGBoost untuk classification dan regression
- **Ensemble Predictions**: Menggabungkan prediksi dari multiple models
- **Interactive CLI**: Menu interaktif untuk semua fungsi

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. API key dikonfigurasi di file `.env`

## Quick Start

### Option 1: Menggunakan Menu Interaktif

Jalankan aplikasi utama:
```bash
python main.py
```

Kemudian ikuti menu:
1. Test API Connection
2. Collect Market Data (optional)
3. Generate Sample Data with Indicators
4. Train ML Models
5. Make Predictions
6. View Model Performance

### Option 2: Step-by-Step Manual

1. **Test API Connection**:
```bash
python data_collector.py
```

2. **Generate Sample Data dengan Technical Indicators**:
```bash
python technical_indicators.py
```

3. **Train Semua Models**:
```bash
python model_trainer.py
```

4. **Make Predictions**:
```bash
python predictor.py
```

## Project Structure

```
criptod/
├── config.py                 # Configuration dan settings
├── utils.py                  # Utility functions
├── data_collector.py         # CoinMarketCap API client
├── technical_indicators.py   # Technical analysis
├── feature_engineering.py    # Feature preparation
├── models/
│   ├── lstm_model.py        # LSTM neural network
│   ├── random_forest_model.py # Random Forest classifier
│   └── xgboost_model.py     # XGBoost model
├── model_trainer.py         # Unified training pipeline
├── predictor.py             # Prediction system
├── main.py                  # Main application
├── requirements.txt         # Dependencies
└── .env                     # API keys
```

## Models

### 1. LSTM (Long Short-Term Memory)
- Deep learning model untuk time-series forecasting
- Multi-layer architecture dengan dropout
- Predicts: Price returns

### 2. Random Forest
- Ensemble classifier untuk market direction
- Feature importance analysis
- Predicts: UP / DOWN / NEUTRAL

### 3. XGBoost
- Gradient boosting untuk classification dan regression
- High performance dan accuracy
- Predicts: Market direction dan price returns

## Technical Indicators

- **Trend**: SMA, EMA, MACD
- **Momentum**: RSI, Stochastic, ROC
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume Ratio
- **Price Action**: Lagged features, Rolling statistics

## API Usage

CoinMarketCap API key sudah dikonfigurasi. Free tier limitations:
- 333 calls/day
- 10,000 calls/month
- Latest data only (historical data requires paid plan)

## Notes

- Sample data digunakan untuk training karena historical data memerlukan paid API plan
- Untuk production, gunakan paid API atau alternative data sources (CoinGecko, Binance, etc.)
- Models disimpan di folder `models/`
- Data disimpan di folder `data/`
- Outputs disimpan di folder `outputs/`

## Performance Metrics

Setelah training, lihat performance report:
- Classification: Accuracy, Precision, Recall, F1-Score
- Regression: RMSE, MAE, R² Score
- Feature importance analysis

## Disclaimer

Sistem ini dibuat untuk tujuan edukasi dan research. Cryptocurrency trading memiliki risiko tinggi. Selalu lakukan research sendiri dan jangan invest lebih dari yang Anda mampu untuk kehilangan.
