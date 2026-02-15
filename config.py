"""
Configuration file for Crypto Market Prediction System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
COINMARKETCAP_API_KEY = os.getenv('COINMARKETCAP_API_KEY')
COINMARKETCAP_BASE_URL = 'https://pro-api.coinmarketcap.com/v1'

# Cryptocurrencies to analyze
CRYPTOCURRENCIES = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'MATIC', 'AVAX']

# Data Collection Settings
DATA_LIMIT = 100  # Number of cryptocurrencies to fetch
HISTORICAL_DAYS = 365  # Days of historical data to collect

# Technical Indicators Settings
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
SMA_PERIODS = [7, 14, 30, 50, 100, 200]
EMA_PERIODS = [12, 26, 50]

# Feature Engineering
LOOKBACK_PERIOD = 60  # Number of past days to use for prediction
PREDICTION_HORIZON = 1  # Days ahead to predict

# Model Training Settings
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# LSTM Model Configuration
LSTM_UNITS = [128, 64, 32]
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001

# Random Forest Configuration
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2

# XGBoost Configuration
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 10
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# Prediction Thresholds
UP_THRESHOLD = 0.02  # 2% increase
DOWN_THRESHOLD = -0.02  # 2% decrease

# Directories
DATA_DIR = 'data'
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)
