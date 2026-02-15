"""
Utility functions for the Crypto Market Prediction System
"""
import logging
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Setup logging
def setup_logger(name, log_file='crypto_ml.log', level=logging.INFO):
    """Setup logger with file and console handlers"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Data validation
def validate_dataframe(df, required_columns):
    """Validate if dataframe has required columns"""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

# Save and load functions
def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

# Date utilities
def get_timestamp():
    """Get current timestamp"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_date_range(days):
    """Get date range for the last N days"""
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=days)
    return start_date, end_date

# Data preprocessing
def remove_outliers(df, column, n_std=3):
    """Remove outliers using standard deviation method"""
    mean = df[column].mean()
    std = df[column].std()
    df_filtered = df[
        (df[column] >= mean - n_std * std) & 
        (df[column] <= mean + n_std * std)
    ]
    return df_filtered

def fill_missing_values(df, method='ffill'):
    """Fill missing values in dataframe"""
    if method == 'ffill':
        return df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'interpolate':
        return df.interpolate(method='linear')
    else:
        return df.fillna(0)

# Performance metrics
def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate various accuracy metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred)
    }
    return metrics

# Display utilities
def print_section(title):
    """Print a formatted section title"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_metrics(metrics, title="Metrics"):
    """Print metrics in a formatted way"""
    print_section(title)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.upper()}: {value:.4f}")
        else:
            print(f"{key.upper()}: {value}")

# Error handling
class CryptoMLError(Exception):
    """Base exception for Crypto ML errors"""
    pass

class APIError(CryptoMLError):
    """Exception for API-related errors"""
    pass

class DataError(CryptoMLError):
    """Exception for data-related errors"""
    pass

class ModelError(CryptoMLError):
    """Exception for model-related errors"""
    pass
