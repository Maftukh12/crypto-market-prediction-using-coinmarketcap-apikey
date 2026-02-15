"""
Feature Engineering for Crypto Price Prediction
Prepares features and labels for machine learning models
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import config
from utils import setup_logger, DataError
import joblib
import os

logger = setup_logger(__name__)

class FeatureEngineer:
    """Feature engineering for cryptocurrency prediction"""
    
    def __init__(self, df):
        """Initialize with dataframe containing price data and indicators"""
        self.df = df.copy()
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.target_column = None
        
    def create_lagged_features(self, columns, lags=[1, 2, 3, 5, 7]):
        """Create lagged features for time series"""
        logger.info(f"Creating lagged features for {len(columns)} columns...")
        
        for col in columns:
            if col in self.df.columns:
                for lag in lags:
                    self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        
        logger.info(f"✓ Created lagged features with lags: {lags}")
        return self
    
    def create_rolling_features(self, columns, windows=[3, 7, 14, 30]):
        """Create rolling window statistics"""
        logger.info(f"Creating rolling features for {len(columns)} columns...")
        
        for col in columns:
            if col in self.df.columns:
                for window in windows:
                    if len(self.df) >= window:
                        self.df[f'{col}_rolling_mean_{window}'] = self.df[col].rolling(window=window).mean()
                        self.df[f'{col}_rolling_std_{window}'] = self.df[col].rolling(window=window).std()
        
        logger.info(f"✓ Created rolling features with windows: {windows}")
        return self
    
    def create_target_classification(self, threshold_up=None, threshold_down=None):
        """
        Create classification target: UP (1), DOWN (-1), NEUTRAL (0)
        Based on next day's price change
        """
        if threshold_up is None:
            threshold_up = config.UP_THRESHOLD
        if threshold_down is None:
            threshold_down = config.DOWN_THRESHOLD
        
        # Calculate next day's return
        self.df['future_return'] = self.df['close'].shift(-1) / self.df['close'] - 1
        
        # Create target labels
        self.df['target_class'] = 0  # NEUTRAL
        self.df.loc[self.df['future_return'] > threshold_up, 'target_class'] = 1  # UP
        self.df.loc[self.df['future_return'] < threshold_down, 'target_class'] = -1  # DOWN
        
        # Distribution
        dist = self.df['target_class'].value_counts()
        logger.info(f"✓ Created classification target:")
        logger.info(f"   UP (1): {dist.get(1, 0)} samples")
        logger.info(f"   NEUTRAL (0): {dist.get(0, 0)} samples")
        logger.info(f"   DOWN (-1): {dist.get(-1, 0)} samples")
        
        self.target_column = 'target_class'
        return self
    
    def create_target_regression(self):
        """Create regression target: next day's price"""
        self.df['target_price'] = self.df['close'].shift(-1)
        self.df['target_return'] = self.df['future_return']
        
        logger.info("✓ Created regression target (next day's price)")
        self.target_column = 'target_return'
        return self
    
    def select_features(self, exclude_columns=None):
        """Select feature columns for modeling"""
        if exclude_columns is None:
            exclude_columns = [
                'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'target_class', 'target_price', 'target_return', 'future_return',
                'symbol', 'name'
            ]
        
        # Get all columns except excluded ones
        self.feature_columns = [col for col in self.df.columns if col not in exclude_columns]
        
        logger.info(f"✓ Selected {len(self.feature_columns)} features")
        return self
    
    def remove_nan(self):
        """Remove rows with NaN values"""
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        removed_rows = initial_rows - len(self.df)
        
        logger.info(f"✓ Removed {removed_rows} rows with NaN values")
        logger.info(f"✓ Remaining rows: {len(self.df)}")
        return self
    
    def scale_features(self, method='minmax'):
        """Scale features using MinMax or Standard scaling"""
        if self.feature_columns is None:
            raise DataError("Feature columns not selected. Call select_features() first.")
        
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        self.df[self.feature_columns] = self.scaler.fit_transform(self.df[self.feature_columns])
        
        logger.info(f"✓ Scaled features using {method} scaling")
        return self
    
    def prepare_train_test_split(self, test_size=None, shuffle=False):
        """
        Split data into train and test sets
        For time series, shuffle should be False
        """
        if test_size is None:
            test_size = config.TEST_SIZE
        
        if self.feature_columns is None or self.target_column is None:
            raise DataError("Features and target not prepared")
        
        X = self.df[self.feature_columns].values
        y = self.df[self.target_column].values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            shuffle=shuffle,
            random_state=config.RANDOM_STATE
        )
        
        logger.info(f"✓ Train/Test split:")
        logger.info(f"   Train: {len(X_train)} samples")
        logger.info(f"   Test: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_lstm_sequences(self, lookback=None):
        """
        Prepare sequences for LSTM model
        Returns 3D arrays (samples, timesteps, features)
        """
        if lookback is None:
            lookback = config.LOOKBACK_PERIOD
        
        if self.feature_columns is None or self.target_column is None:
            raise DataError("Features and target not prepared")
        
        X = self.df[self.feature_columns].values
        y = self.df[self.target_column].values
        
        X_seq = []
        y_seq = []
        
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split
        split_idx = int(len(X_seq) * (1 - config.TEST_SIZE))
        
        X_train = X_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_train = y_seq[:split_idx]
        y_test = y_seq[split_idx:]
        
        logger.info(f"✓ LSTM sequences prepared (lookback={lookback}):")
        logger.info(f"   Train: {X_train.shape}")
        logger.info(f"   Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_scaler(self, filename=None):
        """Save the fitted scaler"""
        if filename is None:
            filename = f"{config.MODELS_DIR}/scaler.pkl"
        
        joblib.dump(self.scaler, filename)
        logger.info(f"✓ Scaler saved to {filename}")
    
    def load_scaler(self, filename=None):
        """Load a fitted scaler"""
        if filename is None:
            filename = f"{config.MODELS_DIR}/scaler.pkl"
        
        if os.path.exists(filename):
            self.scaler = joblib.load(filename)
            logger.info(f"✓ Scaler loaded from {filename}")
        else:
            logger.warning(f"Scaler file not found: {filename}")
    
    def get_dataframe(self):
        """Return the processed dataframe"""
        return self.df
    
    def get_feature_names(self):
        """Return feature column names"""
        return self.feature_columns

def main():
    """Test feature engineering"""
    print("="*60)
    print("  Feature Engineering Test")
    print("="*60)
    
    # Load sample data with indicators
    sample_file = f"{config.DATA_DIR}/sample_with_indicators.csv"
    
    if not os.path.exists(sample_file):
        print(f"\n✗ Sample file not found: {sample_file}")
        print("  Run technical_indicators.py first to generate sample data")
        return
    
    df = pd.read_csv(sample_file)
    print(f"\n1. Loaded sample data: {len(df)} rows, {len(df.columns)} columns")
    
    # Feature engineering
    print("\n2. Engineering features...")
    fe = FeatureEngineer(df)
    
    # Create lagged features for key indicators
    fe.create_lagged_features(['close', 'volume', 'rsi', 'macd'], lags=[1, 2, 3, 7])
    
    # Create rolling features
    fe.create_rolling_features(['close', 'volume'], windows=[7, 14, 30])
    
    # Create targets
    fe.create_target_classification()
    fe.create_target_regression()
    
    # Select features
    fe.select_features()
    
    # Remove NaN
    fe.remove_nan()
    
    print(f"\n3. Final dataset: {len(fe.df)} rows, {len(fe.feature_columns)} features")
    
    # Scale features
    print("\n4. Scaling features...")
    fe.scale_features(method='minmax')
    
    # Prepare splits
    print("\n5. Preparing train/test splits...")
    
    # For classification
    fe.target_column = 'target_class'
    X_train, X_test, y_train, y_test = fe.prepare_train_test_split()
    
    print(f"\n   Classification split:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    
    # For LSTM
    print("\n6. Preparing LSTM sequences...")
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = fe.prepare_lstm_sequences(lookback=30)
    
    # Save scaler
    fe.save_scaler()
    
    print("\n✓ Feature engineering complete!")

if __name__ == "__main__":
    main()
