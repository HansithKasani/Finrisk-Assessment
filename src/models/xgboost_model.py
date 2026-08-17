"""
XGBoost model training, evaluation, and prediction module
"""

import joblib
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


class CreditRiskModel:
    """XGBoost model for credit risk assessment"""
    
    def __init__(self, model_path: str = "Models/credit_risk_xgboost_model.pkl"):
        self.model = None
        self.model_path = Path(model_path)
        self.performance_metrics = {}
        self.threshold = 0.20  # Optimized threshold
        
        # Default hyperparameters
        self.hyperparameters = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              use_smote: bool = True):
        """Train XGBoost model"""
        logger.info("Starting model training...")
        
        # Apply SMOTE to handle class imbalance
        if use_smote:
            logger.info("Applying SMOTE for class balancing")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Train set size: {len(X_train)}")
        
        # Prepare evaluation set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Train model
        self.model = xgb.XGBClassifier(**self.hyperparameters)
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50 if eval_set else None,
            verbose=True
        )
        
        logger.info("Model training completed")
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating model...")
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Calculate metrics
        self.performance_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"Performance Metrics: {self.performance_metrics}")
        
        # Confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{cr}")
        
        return self.performance_metrics
    
    def predict(self, X: np.ndarray, return_probability: bool = False) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        
        if return_probability:
            return predictions, probabilities
        return predictions
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict for a single applicant"""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        # Convert dict to array (assumes features are in correct order)
        X = np.array(list(features.values())).reshape(1, -1)
        
        probability = self.model.predict_proba(X)[0, 1]
        prediction = 1 if probability >= self.threshold else 0
        
        # Determine risk level
        if probability < 0.2:
            risk_level = "low"
        elif probability < 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "prediction": prediction,
            "risk_probability": float(probability),
            "risk_level": risk_level,
            "confidence": float(max(self.model.predict_proba(X)[0]))
        }
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        importance = self.model.feature_importances_
        feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Sort and get top N
        indices = np.argsort(importance)[::-1][:top_n]
        
        return {feature_names[i]: float(importance[i]) for i in indices}
    
    def save(self, path: str = None):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not trained. Nothing to save.")
        
        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path: str = None):
        """Load model from disk"""
        load_path = Path(path) if path else self.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        self.model = joblib.load(load_path)
        logger.info(f"Model loaded from {load_path}")
    
    def set_threshold(self, threshold: float):
        """Set prediction threshold"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        logger.info(f"Prediction threshold set to {threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_type": "XGBoost",
            "hyperparameters": self.hyperparameters,
            "threshold": self.threshold,
            "performance_metrics": self.performance_metrics,
            "model_loaded": self.model is not None
        }


class ModelExplainer:
    """Model explanation utilities (SHAP/LIME)"""
    
    def __init__(self, model):
        self.model = model
        self.explainer = None
    
    def explain_prediction(self, X: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain a single prediction using SHAP"""
        try:
            import shap
            
            if self.explainer is None:
                # Use tree explainer for XGBoost
                self.explainer = shap.TreeExplainer(self.model)
            
            shap_values = self.explainer.shap_values(X[sample_idx:sample_idx+1])
            
            return {
                "base_value": float(self.explainer.expected_value),
                "shap_values": shap_values,
                "prediction": float(self.model.predict_proba(X[sample_idx:sample_idx+1])[0, 1])
            }
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return {}
    
    def get_global_importance(self) -> Dict[str, float]:
        """Get global feature importance from SHAP"""
        try:
            import shap
            
            if self.explainer is None:
                raise ValueError("Explainer not initialized")
            
            # This would require calling on the full dataset
            # For now, return a placeholder
            return {}
        except ImportError:
            logger.warning("SHAP not installed")
            return {}
