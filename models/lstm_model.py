"""
LSTM Model for Cryptocurrency Price Prediction
Deep learning model for time series forecasting
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import config
from utils import setup_logger, calculate_regression_metrics, ModelError
import os

logger = setup_logger(__name__)

class LSTMModel:
    """LSTM Neural Network for crypto price prediction"""
    
    def __init__(self, input_shape):
        """
        Initialize LSTM model
        input_shape: (timesteps, features)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self, units=None, dropout=None, learning_rate=None):
        """Build LSTM architecture"""
        if units is None:
            units = config.LSTM_UNITS
        if dropout is None:
            dropout = config.LSTM_DROPOUT
        if learning_rate is None:
            learning_rate = config.LSTM_LEARNING_RATE
        
        logger.info("Building LSTM model...")
        
        model = Sequential([
            # First LSTM layer
            LSTM(units[0], return_sequences=True, input_shape=self.input_shape),
            Dropout(dropout),
            
            # Second LSTM layer
            LSTM(units[1], return_sequences=True),
            Dropout(dropout),
            
            # Third LSTM layer
            LSTM(units[2], return_sequences=False),
            Dropout(dropout),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(dropout),
            
            Dense(16, activation='relu'),
            
            # Output layer (regression)
            Dense(1, activation='linear')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        logger.info(f"✓ LSTM model built:")
        logger.info(f"   Units: {units}")
        logger.info(f"   Dropout: {dropout}")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Total parameters: {model.count_params():,}")
        
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=None, batch_size=None):
        """Train the LSTM model"""
        if self.model is None:
            raise ModelError("Model not built. Call build_model() first.")
        
        if epochs is None:
            epochs = config.LSTM_EPOCHS
        if batch_size is None:
            batch_size = config.LSTM_BATCH_SIZE
        
        logger.info(f"Training LSTM model...")
        logger.info(f"   Training samples: {len(X_train)}")
        if X_val is not None:
            logger.info(f"   Validation samples: {len(X_val)}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f"{config.MODELS_DIR}/lstm_best.h5",
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✓ Training complete!")
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ModelError("Model not trained")
        
        logger.info("Evaluating LSTM model...")
        
        # Predictions
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        
        # Calculate metrics
        metrics = calculate_regression_metrics(y_test, y_pred)
        
        logger.info("✓ LSTM Model Performance:")
        logger.info(f"   RMSE: {metrics['rmse']:.6f}")
        logger.info(f"   MAE: {metrics['mae']:.6f}")
        logger.info(f"   R² Score: {metrics['r2_score']:.6f}")
        
        return metrics, y_pred
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ModelError("Model not trained")
        
        predictions = self.model.predict(X, verbose=0).flatten()
        return predictions
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f"{config.MODELS_DIR}/lstm_model.h5"
        
        self.model.save(filepath)
        logger.info(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            filepath = f"{config.MODELS_DIR}/lstm_model.h5"
        
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            logger.info(f"✓ Model loaded from {filepath}")
        else:
            raise ModelError(f"Model file not found: {filepath}")
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            raise ModelError("Model not built")
        
        return self.model.summary()

def main():
    """Test LSTM model"""
    print("="*60)
    print("  LSTM Model Test")
    print("="*60)
    
    # Create dummy data for testing
    print("\n1. Creating dummy training data...")
    timesteps = 30
    features = 50
    samples = 1000
    
    X_train = np.random.randn(samples, timesteps, features)
    y_train = np.random.randn(samples)
    
    X_test = np.random.randn(200, timesteps, features)
    y_test = np.random.randn(200)
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    
    # Build model
    print("\n2. Building LSTM model...")
    lstm = LSTMModel(input_shape=(timesteps, features))
    lstm.build_model()
    
    # Show summary
    print("\n3. Model architecture:")
    lstm.get_model_summary()
    
    # Train model
    print("\n4. Training model (this may take a while)...")
    lstm.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
    
    # Evaluate
    print("\n5. Evaluating model...")
    metrics, predictions = lstm.evaluate(X_test, y_test)
    
    # Save model
    print("\n6. Saving model...")
    lstm.save_model()
    
    print("\n✓ LSTM model test complete!")

if __name__ == "__main__":
    main()
