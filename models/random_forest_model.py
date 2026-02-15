"""
Random Forest Model for Cryptocurrency Price Prediction
Classification model for predicting market direction
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import config
from utils import setup_logger, calculate_accuracy_metrics, ModelError
import joblib
import os

logger = setup_logger(__name__)

class RandomForestModel:
    """Random Forest classifier for crypto market direction prediction"""
    
    def __init__(self):
        """Initialize Random Forest model"""
        self.model = None
        self.feature_importance = None
        
    def build_model(self, n_estimators=None, max_depth=None, 
                   min_samples_split=None, min_samples_leaf=None):
        """Build Random Forest classifier"""
        if n_estimators is None:
            n_estimators = config.RF_N_ESTIMATORS
        if max_depth is None:
            max_depth = config.RF_MAX_DEPTH
        if min_samples_split is None:
            min_samples_split = config.RF_MIN_SAMPLES_SPLIT
        if min_samples_leaf is None:
            min_samples_leaf = config.RF_MIN_SAMPLES_LEAF
        
        logger.info("Building Random Forest model...")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        
        logger.info(f"✓ Random Forest model built:")
        logger.info(f"   N estimators: {n_estimators}")
        logger.info(f"   Max depth: {max_depth}")
        logger.info(f"   Min samples split: {min_samples_split}")
        logger.info(f"   Min samples leaf: {min_samples_leaf}")
        
        return self
    
    def train(self, X_train, y_train, feature_names=None):
        """Train the Random Forest model"""
        if self.model is None:
            raise ModelError("Model not built. Call build_model() first.")
        
        logger.info(f"Training Random Forest model...")
        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Features: {X_train.shape[1]}")
        
        # Convert labels to categorical (0, 1, 2) if they are (-1, 0, 1)
        y_train_encoded = y_train.copy()
        if np.min(y_train) == -1:
            y_train_encoded = y_train + 1  # Convert -1,0,1 to 0,1,2
        
        # Train
        self.model.fit(X_train, y_train_encoded)
        
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
        
        logger.info("Evaluating Random Forest model...")
        
        # Encode labels
        y_test_encoded = y_test.copy()
        if np.min(y_test) == -1:
            y_test_encoded = y_test + 1
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Decode predictions back to original labels
        y_pred_decoded = y_pred.copy()
        if np.min(y_test) == -1:
            y_pred_decoded = y_pred - 1
        
        # Calculate metrics
        metrics = calculate_accuracy_metrics(y_test_encoded, y_pred)
        
        logger.info("✓ Random Forest Model Performance:")
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
        
        return metrics, y_pred_decoded
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ModelError("Model not trained")
        
        predictions = self.model.predict(X)
        
        # Decode if needed
        if hasattr(self, '_encoded') and self._encoded:
            predictions = predictions - 1
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ModelError("Model not trained")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f"{config.MODELS_DIR}/random_forest_model.pkl"
        
        joblib.dump(self.model, filepath)
        logger.info(f"✓ Model saved to {filepath}")
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_file = f"{config.MODELS_DIR}/feature_importance.csv"
            self.feature_importance.to_csv(importance_file, index=False)
            logger.info(f"✓ Feature importance saved to {importance_file}")
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            filepath = f"{config.MODELS_DIR}/random_forest_model.pkl"
        
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            logger.info(f"✓ Model loaded from {filepath}")
        else:
            raise ModelError(f"Model file not found: {filepath}")
    
    def get_feature_importance(self):
        """Get feature importance dataframe"""
        return self.feature_importance

def main():
    """Test Random Forest model"""
    print("="*60)
    print("  Random Forest Model Test")
    print("="*60)
    
    # Create dummy data for testing
    print("\n1. Creating dummy training data...")
    samples = 1000
    features = 50
    
    X_train = np.random.randn(samples, features)
    y_train = np.random.choice([-1, 0, 1], size=samples)  # DOWN, NEUTRAL, UP
    
    X_test = np.random.randn(200, features)
    y_test = np.random.choice([-1, 0, 1], size=200)
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   Class distribution: {np.bincount(y_train + 1)}")
    
    # Build model
    print("\n2. Building Random Forest model...")
    rf = RandomForestModel()
    rf.build_model()
    
    # Train model
    print("\n3. Training model...")
    feature_names = [f'feature_{i}' for i in range(features)]
    rf.train(X_train, y_train, feature_names=feature_names)
    
    # Evaluate
    print("\n4. Evaluating model...")
    metrics, predictions = rf.evaluate(X_test, y_test)
    
    # Save model
    print("\n5. Saving model...")
    rf.save_model()
    
    print("\n✓ Random Forest model test complete!")

if __name__ == "__main__":
    main()
