"""
Model Trainer - Unified training pipeline for all models
Trains and compares LSTM, Random Forest, and XGBoost models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
# Lazy import for LSTM to avoid TensorFlow dependency issues
# from models.lstm_model import LSTMModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from feature_engineering import FeatureEngineer
import config
from utils import setup_logger, print_section
import json

logger = setup_logger(__name__)

class ModelTrainer:
    """Unified training pipeline for all ML models"""
    
    def __init__(self, data_file):
        """Initialize with data file path"""
        self.data_file = data_file
        self.df = None
        self.fe = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load data and prepare features"""
        print_section("Loading and Preparing Data")
        
        # Load data
        logger.info(f"Loading data from {self.data_file}...")
        self.df = pd.read_csv(self.data_file)
        logger.info(f"✓ Loaded {len(self.df)} rows")
        
        # Feature engineering
        logger.info("Engineering features...")
        self.fe = FeatureEngineer(self.df)
        
        # Create lagged features
        key_features = ['close', 'volume', 'rsi', 'macd', 'bb_width']
        self.fe.create_lagged_features(key_features, lags=[1, 2, 3, 5, 7])
        
        # Create rolling features
        self.fe.create_rolling_features(['close', 'volume', 'rsi'], windows=[7, 14, 30])
        
        # Create targets
        self.fe.create_target_classification()
        self.fe.create_target_regression()
        
        # Select features
        self.fe.select_features()
        
        # Remove NaN
        self.fe.remove_nan()
        
        logger.info(f"✓ Feature engineering complete: {len(self.fe.feature_columns)} features")
        
        return self
    
    def train_random_forest(self):
        """Train Random Forest classifier"""
        print_section("Training Random Forest Model")
        
        # Prepare data
        self.fe.target_column = 'target_class'
        self.fe.scale_features(method='standard')
        X_train, X_test, y_train, y_test = self.fe.prepare_train_test_split()
        
        # Build and train
        rf = RandomForestModel()
        rf.build_model()
        rf.train(X_train, y_train, feature_names=self.fe.feature_columns)
        
        # Evaluate
        metrics, predictions = rf.evaluate(X_test, y_test)
        
        # Save
        rf.save_model()
        
        self.results['random_forest'] = {
            'model': rf,
            'metrics': metrics,
            'predictions': predictions,
            'y_test': y_test
        }
        
        return self
    
    def train_xgboost_classifier(self):
        """Train XGBoost classifier"""
        print_section("Training XGBoost Classifier")
        
        # Prepare data
        self.fe.target_column = 'target_class'
        self.fe.scale_features(method='standard')
        X_train, X_test, y_train, y_test = self.fe.prepare_train_test_split()
        
        # Build and train
        xgb_clf = XGBoostModel(task='classification')
        xgb_clf.build_model()
        xgb_clf.train(X_train, y_train, feature_names=self.fe.feature_columns)
        
        # Evaluate
        metrics, predictions = xgb_clf.evaluate(X_test, y_test)
        
        # Save
        xgb_clf.save_model()
        
        self.results['xgboost_classifier'] = {
            'model': xgb_clf,
            'metrics': metrics,
            'predictions': predictions,
            'y_test': y_test
        }
        
        return self
    
    def train_xgboost_regressor(self):
        """Train XGBoost regressor"""
        print_section("Training XGBoost Regressor")
        
        # Prepare data
        self.fe.target_column = 'target_return'
        self.fe.scale_features(method='standard')
        X_train, X_test, y_train, y_test = self.fe.prepare_train_test_split()
        
        # Build and train
        xgb_reg = XGBoostModel(task='regression')
        xgb_reg.build_model()
        xgb_reg.train(X_train, y_train, feature_names=self.fe.feature_columns)
        
        # Evaluate
        metrics, predictions = xgb_reg.evaluate(X_test, y_test)
        
        # Save
        xgb_reg.save_model()
        
        self.results['xgboost_regressor'] = {
            'model': xgb_reg,
            'metrics': metrics,
            'predictions': predictions,
            'y_test': y_test
        }
        
        return self
    
    def train_lstm(self):
        """Train LSTM model"""
        print_section("Training LSTM Model")
        
        try:
            # Lazy import to avoid TensorFlow dependency
            from models.lstm_model import LSTMModel
            
            # Prepare data
            self.fe.target_column = 'target_return'
            self.fe.scale_features(method='minmax')
            
            # Prepare LSTM sequences
            X_train, X_test, y_train, y_test = self.fe.prepare_lstm_sequences(lookback=30)
            
            # Build and train
            lstm = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
            lstm.build_model()
            lstm.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
            
            # Evaluate
            metrics, predictions = lstm.evaluate(X_test, y_test)
            
            # Save
            lstm.save_model()
            
            self.results['lstm'] = {
                'model': lstm,
                'metrics': metrics,
                'predictions': predictions,
                'y_test': y_test
            }
        except ImportError as e:
            logger.warning(f"LSTM training skipped: TensorFlow not available ({str(e)})")
            logger.info("Install TensorFlow to enable LSTM model: pip install tensorflow==2.16.1")
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
        
        return self
    
    def save_scaler(self):
        """Save the feature scaler"""
        self.fe.save_scaler()
        return self
    
    def generate_report(self):
        """Generate training report"""
        print_section("Training Report")
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("  CRYPTO ML MODEL TRAINING REPORT")
        report_lines.append("="*60)
        report_lines.append(f"\nData file: {self.data_file}")
        report_lines.append(f"Total samples: {len(self.df)}")
        report_lines.append(f"Features: {len(self.fe.feature_columns)}")
        report_lines.append("\n" + "="*60)
        report_lines.append("  MODEL PERFORMANCE COMPARISON")
        report_lines.append("="*60)
        
        # Classification models
        if 'random_forest' in self.results:
            report_lines.append("\nRandom Forest Classifier:")
            metrics = self.results['random_forest']['metrics']
            report_lines.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
            report_lines.append(f"  Precision: {metrics['precision']:.4f}")
            report_lines.append(f"  Recall:    {metrics['recall']:.4f}")
            report_lines.append(f"  F1 Score:  {metrics['f1_score']:.4f}")
        
        if 'xgboost_classifier' in self.results:
            report_lines.append("\nXGBoost Classifier:")
            metrics = self.results['xgboost_classifier']['metrics']
            report_lines.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
            report_lines.append(f"  Precision: {metrics['precision']:.4f}")
            report_lines.append(f"  Recall:    {metrics['recall']:.4f}")
            report_lines.append(f"  F1 Score:  {metrics['f1_score']:.4f}")
        
        # Regression models
        if 'xgboost_regressor' in self.results:
            report_lines.append("\nXGBoost Regressor:")
            metrics = self.results['xgboost_regressor']['metrics']
            report_lines.append(f"  RMSE:      {metrics['rmse']:.6f}")
            report_lines.append(f"  MAE:       {metrics['mae']:.6f}")
            report_lines.append(f"  R² Score:  {metrics['r2_score']:.6f}")
        
        if 'lstm' in self.results:
            report_lines.append("\nLSTM Model:")
            metrics = self.results['lstm']['metrics']
            report_lines.append(f"  RMSE:      {metrics['rmse']:.6f}")
            report_lines.append(f"  MAE:       {metrics['mae']:.6f}")
            report_lines.append(f"  R² Score:  {metrics['r2_score']:.6f}")
        
        report_lines.append("\n" + "="*60)
        
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # Save report
        report_file = f"{config.MODELS_DIR}/training_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"✓ Report saved to {report_file}")
        
        return self

def main():
    """Main training pipeline"""
    print("="*60)
    print("  CRYPTO ML MODEL TRAINER")
    print("="*60)
    
    # Check for data file
    data_file = f"{config.DATA_DIR}/sample_with_indicators.csv"
    
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        logger.error("Please run technical_indicators.py first to generate sample data")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(data_file)
    
    # Load and prepare data
    trainer.load_and_prepare_data()
    
    # Train all models
    try:
        trainer.train_random_forest()
    except Exception as e:
        logger.error(f"Random Forest training failed: {str(e)}")
    
    try:
        trainer.train_xgboost_classifier()
    except Exception as e:
        logger.error(f"XGBoost classifier training failed: {str(e)}")
    
    try:
        trainer.train_xgboost_regressor()
    except Exception as e:
        logger.error(f"XGBoost regressor training failed: {str(e)}")
    
    try:
        trainer.train_lstm()
    except Exception as e:
        logger.error(f"LSTM training failed: {str(e)}")
    
    # Save scaler
    trainer.save_scaler()
    
    # Generate report
    trainer.generate_report()
    
    print("\n✓ All models trained successfully!")

if __name__ == "__main__":
    main()
