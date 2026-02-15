"""
XGBoost Model for Cryptocurrency Price Prediction
Gradient boosting model for classification and regression
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import config
from utils import setup_logger, calculate_accuracy_metrics, calculate_regression_metrics, ModelError
import joblib
import os

logger = setup_logger(__name__)

class XGBoostModel:
    """XGBoost model for crypto prediction"""
    
    def __init__(self, task='classification'):
        """
        Initialize XGBoost model
        task: 'classification' or 'regression'
        """
        self.task = task
        self.model = None
        self.feature_importance = None
        
    def build_model(self, n_estimators=None, max_depth=None, 
                   learning_rate=None, subsample=None, colsample_bytree=None):
        """Build XGBoost model"""
        if n_estimators is None:
            n_estimators = config.XGB_N_ESTIMATORS
        if max_depth is None:
            max_depth = config.XGB_MAX_DEPTH
        if learning_rate is None:
            learning_rate = config.XGB_LEARNING_RATE
        if subsample is None:
            subsample = config.XGB_SUBSAMPLE
        if colsample_bytree is None:
            colsample_bytree = config.XGB_COLSAMPLE_BYTREE
        
        logger.info(f"Building XGBoost model ({self.task})...")
        
        if self.task == 'classification':
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        else:  # regression
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric='rmse'
            )
        
        logger.info(f"✓ XGBoost model built:")
        logger.info(f"   N estimators: {n_estimators}")
        logger.info(f"   Max depth: {max_depth}")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Subsample: {subsample}")
        logger.info(f"   Colsample bytree: {colsample_bytree}")
        
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train the XGBoost model"""
        if self.model is None:
            raise ModelError("Model not built. Call build_model() first.")
        
        logger.info(f"Training XGBoost model...")
        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Features: {X_train.shape[1]}")
        
        # Encode labels for classification
        y_train_encoded = y_train.copy()
        if self.task == 'classification' and np.min(y_train) == -1:
            y_train_encoded = y_train + 1  # Convert -1,0,1 to 0,1,2
        
        # Prepare validation data
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_encoded = y_val.copy()
            if self.task == 'classification' and np.min(y_val) == -1:
                y_val_encoded = y_val + 1
            eval_set = [(X_val, y_val_encoded)]
        
        # Train
        self.model.fit(
            X_train, y_train_encoded,
            eval_set=eval_set,
            verbose=False
        )
        
        # Feature importance
        if feature_names is not None:
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.info("✓ Training complete!")
        
        # Show top features
        if self.feature_importance is not None:
            logger.info("\n   Top 10 most important features:")
            for idx, row in self.feature_importance.head(10).iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ModelError("Model not trained")
        
        logger.info(f"Evaluating XGBoost model ({self.task})...")
        
        if self.task == 'classification':
            # Encode labels
            y_test_encoded = y_test.copy()
            if np.min(y_test) == -1:
                y_test_encoded = y_test + 1
            
            # Predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = calculate_accuracy_metrics(y_test_encoded, y_pred)
            
            logger.info("✓ XGBoost Model Performance:")
            logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"   Precision: {metrics['precision']:.4f}")
            logger.info(f"   Recall: {metrics['recall']:.4f}")
            logger.info(f"   F1 Score: {metrics['f1_score']:.4f}")
            
            # Classification report
            target_names = ['DOWN', 'NEUTRAL', 'UP'] if np.min(y_test) == -1 else ['Class 0', 'Class 1', 'Class 2']
            logger.info("\n   Classification Report:")
            logger.info("\n" + classification_report(y_test_encoded, y_pred, target_names=target_names))
            
            # Confusion matrix
            cm = confusion_matrix(y_test_encoded, y_pred)
            logger.info("   Confusion Matrix:")
            logger.info(f"\n{cm}")
            
            # Decode predictions
            y_pred_decoded = y_pred - 1 if np.min(y_test) == -1 else y_pred
            
            return metrics, y_pred_decoded
            
        else:  # regression
            # Predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = calculate_regression_metrics(y_test, y_pred)
            
            logger.info("✓ XGBoost Model Performance:")
            logger.info(f"   RMSE: {metrics['rmse']:.6f}")
            logger.info(f"   MAE: {metrics['mae']:.6f}")
            logger.info(f"   R² Score: {metrics['r2_score']:.6f}")
            
            return metrics, y_pred
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ModelError("Model not trained")
        
        predictions = self.model.predict(X)
        
        # Decode classification predictions
        if self.task == 'classification' and hasattr(self, '_encoded'):
            predictions = predictions - 1
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities (classification only)"""
        if self.model is None:
            raise ModelError("Model not trained")
        
        if self.task != 'classification':
            raise ModelError("predict_proba only available for classification")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f"{config.MODELS_DIR}/xgboost_{self.task}_model.pkl"
        
        joblib.dump(self.model, filepath)
        logger.info(f"✓ Model saved to {filepath}")
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_file = f"{config.MODELS_DIR}/xgboost_feature_importance.csv"
            self.feature_importance.to_csv(importance_file, index=False)
            logger.info(f"✓ Feature importance saved to {importance_file}")
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            filepath = f"{config.MODELS_DIR}/xgboost_{self.task}_model.pkl"
        
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            logger.info(f"✓ Model loaded from {filepath}")
        else:
            raise ModelError(f"Model file not found: {filepath}")
    
    def get_feature_importance(self):
        """Get feature importance dataframe"""
        return self.feature_importance

def main():
    """Test XGBoost model"""
    print("="*60)
    print("  XGBoost Model Test")
    print("="*60)
    
    # Create dummy data for testing
    print("\n1. Creating dummy training data...")
    samples = 1000
    features = 50
    
    X_train = np.random.randn(samples, features)
    y_train_class = np.random.choice([-1, 0, 1], size=samples)
    y_train_reg = np.random.randn(samples)
    
    X_test = np.random.randn(200, features)
    y_test_class = np.random.choice([-1, 0, 1], size=200)
    y_test_reg = np.random.randn(200)
    
    feature_names = [f'feature_{i}' for i in range(features)]
    
    # Test Classification
    print("\n" + "="*60)
    print("  Testing Classification Model")
    print("="*60)
    
    print("\n2. Building XGBoost classifier...")
    xgb_clf = XGBoostModel(task='classification')
    xgb_clf.build_model()
    
    print("\n3. Training classifier...")
    xgb_clf.train(X_train, y_train_class, feature_names=feature_names)
    
    print("\n4. Evaluating classifier...")
    metrics_clf, predictions_clf = xgb_clf.evaluate(X_test, y_test_class)
    
    print("\n5. Saving classifier...")
    xgb_clf.save_model()
    
    # Test Regression
    print("\n" + "="*60)
    print("  Testing Regression Model")
    print("="*60)
    
    print("\n6. Building XGBoost regressor...")
    xgb_reg = XGBoostModel(task='regression')
    xgb_reg.build_model()
    
    print("\n7. Training regressor...")
    xgb_reg.train(X_train, y_train_reg, feature_names=feature_names)
    
    print("\n8. Evaluating regressor...")
    metrics_reg, predictions_reg = xgb_reg.evaluate(X_test, y_test_reg)
    
    print("\n9. Saving regressor...")
    xgb_reg.save_model()
    
    print("\n✓ XGBoost model test complete!")

if __name__ == "__main__":
    main()
