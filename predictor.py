"""
Predictor - Real-time cryptocurrency market prediction
Uses trained models to predict market direction
"""
import pandas as pd
import numpy as np
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
# Lazy import for LSTM
# from models.lstm_model import LSTMModel
from data_collector import CoinMarketCapAPI
from technical_indicators import TechnicalIndicators
from feature_engineering import FeatureEngineer
import config
from utils import setup_logger, print_section
from datetime import datetime
import os

logger = setup_logger(__name__)

class CryptoPredictor:
    """Real-time cryptocurrency market predictor"""
    
    def __init__(self):
        """Initialize predictor with trained models"""
        self.api = CoinMarketCapAPI()
        self.models = {}
        self.scaler = None
        self.load_models()
        
    def load_models(self):
        """Load all trained models"""
        logger.info("Loading trained models...")
        
        try:
            # Random Forest
            rf = RandomForestModel()
            rf.load_model()
            self.models['random_forest'] = rf
            logger.info("✓ Random Forest loaded")
        except Exception as e:
            logger.warning(f"Could not load Random Forest: {str(e)}")
        
        try:
            # XGBoost Classifier
            xgb_clf = XGBoostModel(task='classification')
            xgb_clf.load_model()
            self.models['xgboost_classifier'] = xgb_clf
            logger.info("✓ XGBoost Classifier loaded")
        except Exception as e:
            logger.warning(f"Could not load XGBoost Classifier: {str(e)}")
        
        try:
            # XGBoost Regressor
            xgb_reg = XGBoostModel(task='regression')
            xgb_reg.load_model()
            self.models['xgboost_regressor'] = xgb_reg
            logger.info("✓ XGBoost Regressor loaded")
        except Exception as e:
            logger.warning(f"Could not load XGBoost Regressor: {str(e)}")
        
        try:
            # LSTM
            from models.lstm_model import LSTMModel
            lstm = LSTMModel(input_shape=(30, 50))  # Will be adjusted based on actual data
            lstm.load_model()
            self.models['lstm'] = lstm
            logger.info("✓ LSTM loaded")
        except ImportError:
            logger.warning("LSTM not available: TensorFlow not installed")
        except Exception as e:
            logger.warning(f"Could not load LSTM: {str(e)}")
        
        # Load scaler
        try:
            from feature_engineering import FeatureEngineer
            fe = FeatureEngineer(pd.DataFrame())
            fe.load_scaler()
            self.scaler = fe.scaler
            logger.info("✓ Scaler loaded")
        except Exception as e:
            logger.warning(f"Could not load scaler: {str(e)}")
        
        if not self.models:
            raise Exception("No models loaded! Please train models first.")
    
    def get_latest_data_with_indicators(self, symbol='BTC'):
        """
        Get latest market data with technical indicators
        Note: This is a simplified version. In production, you'd need historical data.
        """
        logger.info(f"Fetching latest data for {symbol}...")
        
        # For demonstration, we'll use sample data
        # In production, you'd fetch real historical data
        
        # This is a placeholder - you would need to:
        # 1. Fetch historical OHLCV data (requires paid API or alternative source)
        # 2. Calculate technical indicators
        # 3. Prepare features
        
        logger.warning("Using sample data for demonstration")
        
        # Load sample data if available
        sample_file = f"{config.DATA_DIR}/sample_with_indicators.csv"
        if os.path.exists(sample_file):
            df = pd.read_csv(sample_file)
            return df.tail(100)  # Return last 100 rows
        else:
            raise Exception("Sample data not found. Run technical_indicators.py first.")
    
    def prepare_features(self, df):
        """Prepare features from raw data"""
        logger.info("Preparing features...")
        
        fe = FeatureEngineer(df)
        
        # Create lagged features
        key_features = ['close', 'volume', 'rsi', 'macd', 'bb_width']
        fe.create_lagged_features(key_features, lags=[1, 2, 3, 5, 7])
        
        # Create rolling features
        fe.create_rolling_features(['close', 'volume', 'rsi'], windows=[7, 14, 30])
        
        # Create targets (needed for feature selection)
        fe.create_target_classification()
        fe.create_target_regression()
        
        # Select features
        fe.select_features()
        
        # Remove NaN
        fe.remove_nan()
        
        return fe
    
    def predict_single(self, symbol='BTC'):
        """Make prediction for a single cryptocurrency"""
        print_section(f"Predicting {symbol}")
        
        # Get data
        df = self.get_latest_data_with_indicators(symbol)
        
        # Prepare features
        fe = self.prepare_features(df)
        
        # Get latest features
        X_latest = fe.df[fe.feature_columns].iloc[-1:].values
        
        # Scale features
        if self.scaler is not None:
            X_latest_scaled = self.scaler.transform(X_latest)
        else:
            X_latest_scaled = X_latest
        
        # Make predictions
        predictions = {}
        
        # Random Forest
        if 'random_forest' in self.models:
            pred = self.models['random_forest'].predict(X_latest_scaled)[0]
            proba = self.models['random_forest'].predict_proba(X_latest_scaled)[0]
            predictions['random_forest'] = {
                'direction': self._decode_direction(pred),
                'confidence': max(proba)
            }
        
        # XGBoost Classifier
        if 'xgboost_classifier' in self.models:
            pred = self.models['xgboost_classifier'].predict(X_latest_scaled)[0]
            proba = self.models['xgboost_classifier'].predict_proba(X_latest_scaled)[0]
            predictions['xgboost_classifier'] = {
                'direction': self._decode_direction(pred),
                'confidence': max(proba)
            }
        
        # XGBoost Regressor
        if 'xgboost_regressor' in self.models:
            pred = self.models['xgboost_regressor'].predict(X_latest_scaled)[0]
            predictions['xgboost_regressor'] = {
                'return': pred,
                'direction': 'UP' if pred > 0.02 else ('DOWN' if pred < -0.02 else 'NEUTRAL')
            }
        
        # Ensemble prediction
        ensemble = self._ensemble_prediction(predictions)
        
        # Display results
        self._display_predictions(symbol, predictions, ensemble)
        
        return predictions, ensemble
    
    def _decode_direction(self, pred):
        """Decode prediction to direction"""
        if pred == 1:
            return 'UP'
        elif pred == -1:
            return 'DOWN'
        else:
            return 'NEUTRAL'
    
    def _ensemble_prediction(self, predictions):
        """Combine predictions from multiple models"""
        directions = []
        confidences = []
        
        for model_name, pred in predictions.items():
            if 'direction' in pred:
                directions.append(pred['direction'])
                if 'confidence' in pred:
                    confidences.append(pred['confidence'])
        
        # Majority vote
        if directions:
            from collections import Counter
            direction_counts = Counter(directions)
            ensemble_direction = direction_counts.most_common(1)[0][0]
            ensemble_confidence = np.mean(confidences) if confidences else 0.5
        else:
            ensemble_direction = 'NEUTRAL'
            ensemble_confidence = 0.5
        
        return {
            'direction': ensemble_direction,
            'confidence': ensemble_confidence,
            'agreement': len(set(directions)) == 1 if directions else False
        }
    
    def _display_predictions(self, symbol, predictions, ensemble):
        """Display prediction results"""
        print(f"\n{'='*60}")
        print(f"  PREDICTIONS FOR {symbol}")
        print(f"{'='*60}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for model_name, pred in predictions.items():
            print(f"{model_name.upper()}:")
            if 'direction' in pred:
                print(f"  Direction: {pred['direction']}")
            if 'confidence' in pred:
                print(f"  Confidence: {pred['confidence']:.2%}")
            if 'return' in pred:
                print(f"  Expected Return: {pred['return']:.2%}")
            print()
        
        print(f"{'='*60}")
        print(f"  ENSEMBLE PREDICTION")
        print(f"{'='*60}")
        print(f"Direction: {ensemble['direction']}")
        print(f"Confidence: {ensemble['confidence']:.2%}")
        print(f"Model Agreement: {'Yes' if ensemble['agreement'] else 'No'}")
        print(f"{'='*60}\n")
    
    def predict_multiple(self, symbols=None):
        """Make predictions for multiple cryptocurrencies"""
        if symbols is None:
            symbols = config.CRYPTOCURRENCIES[:5]
        
        results = {}
        
        for symbol in symbols:
            try:
                predictions, ensemble = self.predict_single(symbol)
                results[symbol] = {
                    'predictions': predictions,
                    'ensemble': ensemble
                }
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {str(e)}")
        
        return results

def main():
    """Test predictor"""
    print("="*60)
    print("  CRYPTO MARKET PREDICTOR")
    print("="*60)
    
    try:
        # Initialize predictor
        predictor = CryptoPredictor()
        
        # Make prediction
        predictor.predict_single('BTC')
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error("Make sure to train models first by running model_trainer.py")

if __name__ == "__main__":
    main()
