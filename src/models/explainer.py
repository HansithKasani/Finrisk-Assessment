"""
Model Explainability Module for Credit Risk Assessment
Implements SHAP and LIME for Explainable AI (XAI)
Week 6: Transparency & Intelligence
"""

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import joblib
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskExplainer:
    """
    Explainable AI wrapper for credit risk models.
    Implements SHAP and LIME for model interpretability.
    Week 6 Requirements: Summary Plot, Force Plot, Reasoning Report
    """
    
    def __init__(self, model, feature_names: List[str], X_train: Optional[np.ndarray] = None):
        """
        Initialize explainer with trained model
        
        Args:
            model: Trained XGBoost/LightGBM model
            feature_names: List of feature names
            X_train: Training data for SHAP background (optional, uses sample if None)
        """
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        
        # Initialize SHAP explainer
        logger.info("Initializing SHAP explainer...")
        if X_train is not None:
            # Use a sample for faster computation
            background = shap.sample(X_train, min(100, len(X_train)))
        else:
            background = None
            
        try:
            # Try TreeExplainer for tree-based models (XGBoost, LightGBM)
            self.shap_explainer = shap.TreeExplainer(model)
            logger.info("SHAP TreeExplainer initialized successfully")
        except:
            # Fallback to KernelExplainer
            if background is not None:
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba, 
                    background
                )
                logger.info("SHAP KernelExplainer initialized successfully")
            else:
                self.shap_explainer = None
                logger.warning("Could not initialize SHAP explainer")
        
        # Initialize LIME explainer
        if X_train is not None:
            logger.info("Initializing LIME explainer...")
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                feature_names=feature_names,
                class_names=['Repay', 'Default'],
                mode='classification',
                random_state=42
            )
            logger.info("LIME explainer initialized successfully")
        else:
            self.lime_explainer = None
            logger.warning("LIME explainer not initialized (no training data)")
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for given instances
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            SHAP values array
        """
        if self.shap_explainer is None:
            logger.error("SHAP explainer not initialized")
            return None
        
        logger.info(f"Calculating SHAP values for {len(X)} instances...")
        shap_values = self.shap_explainer.shap_values(X)
        
        # Handle multi-output (for binary classification, use positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class (Default)
        
        return shap_values
    
    def plot_global_summary(self, X: np.ndarray, save_path: Optional[str] = None):
        """
        Generate global SHAP summary plot
        Week 6 Task: Summary Plot showing which features most influence the model
        
        Args:
            X: Feature matrix
            save_path: Path to save plot (optional)
        """
        logger.info("Generating SHAP global summary plot...")
        
        shap_values = self.get_shap_values(X)
        
        if shap_values is None:
            logger.error("Cannot generate summary plot without SHAP values")
            return
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=self.feature_names,
            show=False,
            max_display=20
        )
        plt.title("SHAP Global Feature Importance - Top 20 Features", fontsize=14, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plot saved to {save_path}")
        
        plt.close()
        return plt.gcf()
    
    def plot_force_plot(self, X_instance: np.ndarray, instance_index: int = 0, 
                       save_path: Optional[str] = None):
        """
        Generate SHAP force plot for a single prediction
        Week 6 Task: Force Plot showing how specific profile pushed risk score up/down
        
        Args:
            X_instance: Single instance features (1D array)
            instance_index: Index for labeling
            save_path: Path to save plot (optional)
        """
        logger.info(f"Generating SHAP force plot for instance {instance_index}...")
        
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        shap_values = self.get_shap_values(X_instance)
        
        if shap_values is None:
            logger.error("Cannot generate force plot without SHAP values")
            return None
        
        # Get base value (expected value)
        if hasattr(self.shap_explainer, 'expected_value'):
            base_value = self.shap_explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
        else:
            base_value = 0.0
        
        # Generate force plot
        shap.force_plot(
            base_value,
            shap_values[0],
            X_instance[0],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force plot saved to {save_path}")
        
        return plt.gcf()
    
    def get_top_features(self, shap_values: np.ndarray, X_instance: np.ndarray, 
                        top_n: int = 10) -> List[Tuple[str, float, float]]:
        """
        Get top N features by absolute SHAP value impact
        
        Args:
            shap_values: SHAP values for single instance
            X_instance: Feature values for instance
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, shap_value, feature_value) tuples
        """
        # Get absolute SHAP values for ranking
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        
        top_features = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            shap_val = shap_values[idx]
            feature_val = X_instance[idx]
            top_features.append((feature_name, shap_val, feature_val))
        
        return top_features
    
    def generate_reasoning_report(self, X_instance: np.ndarray, 
                                 prediction_proba: float,
                                 threshold: float = 0.5) -> str:
        """
        Generate human-readable reasoning report for a prediction
        Week 6 Task: Convert SHAP values into simple English text
        
        Args:
            X_instance: Single instance features
            prediction_proba: Predicted probability of default
            threshold: Decision threshold
            
        Returns:
            Human-readable explanation string
        """
        logger.info("Generating reasoning report...")
        
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.get_shap_values(X_instance)
        
        if shap_values is None:
            return "Unable to generate explanation (SHAP values not available)"
        
        # Get prediction
        decision = "REJECTED" if prediction_proba >= threshold else "APPROVED"
        risk_level = self._get_risk_level(prediction_proba)
        
        # Get top features
        top_features = self.get_top_features(shap_values[0], X_instance[0], top_n=5)
        
        # Build report
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║           CREDIT RISK ASSESSMENT - REASONING REPORT                ║
╚════════════════════════════════════════════════════════════════════╝

DECISION: {decision}
Default Probability: {prediction_proba:.1%}
Risk Level: {risk_level}
Decision Threshold: {threshold:.1%}

────────────────────────────────────────────────────────────────────

KEY FACTORS INFLUENCING THIS DECISION:
"""
        
        for i, (feature_name, shap_val, feature_val) in enumerate(top_features, 1):
            impact = "INCREASED" if shap_val > 0 else "DECREASED"
            impact_magnitude = abs(shap_val)
            
            # Convert feature name to readable format
            readable_name = self._format_feature_name(feature_name)
            
            # Determine magnitude description
            if impact_magnitude > 0.1:
                magnitude_desc = "Strongly"
            elif impact_magnitude > 0.05:
                magnitude_desc = "Moderately"
            else:
                magnitude_desc = "Slightly"
            
            report += f"\n{i}. {readable_name}: {feature_val:.2f}"
            report += f"\n   → {magnitude_desc} {impact} risk by {abs(shap_val):.3f}"
        
        # Add summary
        report += "\n\n" + "─" * 68 + "\n"
        report += "\nSUMMARY:\n"
        
        if decision == "REJECTED":
            report += "⚠ Application REJECTED due to high default probability.\n"
            report += f"Primary concerns: {self._get_primary_concerns(top_features)}\n"
        else:
            report += "✓ Application APPROVED - acceptable risk level.\n"
            report += f"Positive indicators: {self._get_positive_indicators(top_features)}\n"
        
        report += "\n" + "─" * 68
        report += "\n\nNOTE: This explanation is generated using SHAP (SHapley Additive"
        report += "\nexPlanations) values, which measure each feature's contribution to"
        report += "\nthe final prediction. Positive values increase default risk, while"
        report += "\nnegative values decrease it."
        report += "\n\n" + "═" * 68
        
        return report
    
    def explain_with_lime(self, X_instance: np.ndarray, num_features: int = 10):
        """
        Generate LIME explanation for a single instance
        
        Args:
            X_instance: Single instance features
            num_features: Number of features to include in explanation
            
        Returns:
            LIME explanation object
        """
        if self.lime_explainer is None:
            logger.error("LIME explainer not initialized")
            return None
        
        logger.info("Generating LIME explanation...")
        
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Generate LIME explanation
        explanation = self.lime_explainer.explain_instance(
            X_instance[0],
            self.model.predict_proba,
            num_features=num_features
        )
        
        return explanation
    
    def _get_risk_level(self, probability: float) -> str:
        """Classify risk level based on probability"""
        if probability >= 0.7:
            return "🔴 VERY HIGH"
        elif probability >= 0.5:
            return "🟠 HIGH"
        elif probability >= 0.3:
            return "🟡 MEDIUM"
        elif probability >= 0.15:
            return "🟢 LOW"
        else:
            return "✅ VERY LOW"
    
    def _format_feature_name(self, feature_name: str) -> str:
        """Convert feature name to readable format"""
        # Replace underscores with spaces and title case
        readable = feature_name.replace('_', ' ').title()
        
        # Common abbreviations
        replacements = {
            'Amt': 'Amount',
            'Cnt': 'Count',
            'Ext Source': 'External Source',
            'Dti': 'Debt-to-Income',
            'Sk Id': 'ID'
        }
        
        for abbr, full in replacements.items():
            readable = readable.replace(abbr, full)
        
        return readable
    
    def _get_primary_concerns(self, top_features: List[Tuple]) -> str:
        """Extract primary concerns from negative features"""
        concerns = []
        for feature_name, shap_val, _ in top_features[:3]:
            if shap_val > 0:  # Increases risk
                concerns.append(self._format_feature_name(feature_name))
        
        if not concerns:
            return "Multiple minor risk factors"
        
        return ", ".join(concerns)
    
    def _get_positive_indicators(self, top_features: List[Tuple]) -> str:
        """Extract positive indicators from features"""
        positives = []
        for feature_name, shap_val, _ in top_features[:3]:
            if shap_val < 0:  # Decreases risk
                positives.append(self._format_feature_name(feature_name))
        
        if not positives:
            return "Overall profile within acceptable parameters"
        
        return ", ".join(positives)
    
    def save(self, filepath: str):
        """Save explainer configuration"""
        config = {
            'feature_names': self.feature_names,
        }
        joblib.dump(config, filepath)
        logger.info(f"Explainer config saved to {filepath}")
    
    @staticmethod
    def load(filepath: str, model, X_train: Optional[np.ndarray] = None):
        """Load explainer configuration"""
        config = joblib.load(filepath)
        logger.info(f"Explainer config loaded from {filepath}")
        return CreditRiskExplainer(
            model=model,
            feature_names=config['feature_names'],
            X_train=X_train
        )


def create_explainer_for_model(model_path: str, feature_names: List[str], 
                              X_train: Optional[np.ndarray] = None) -> CreditRiskExplainer:
    """
    Convenience function to create explainer from saved model
    
    Args:
        model_path: Path to saved model (.pkl)
        feature_names: List of feature names
        X_train: Training data sample
        
    Returns:
        Initialized CreditRiskExplainer
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    explainer = CreditRiskExplainer(
        model=model,
        feature_names=feature_names,
        X_train=X_train
    )
    
    return explainer


if __name__ == "__main__":
    print("Credit Risk Explainer Module")
    print("=============================")
    print("\nFeatures:")
    print("- SHAP global summary plots")
    print("- SHAP force plots for individual predictions")
    print("- LIME explanations")
    print("- Human-readable reasoning reports")
    print("\nUsage:")
    print("from src.models.explainer import CreditRiskExplainer")
    print("\nexplainer = CreditRiskExplainer(model, feature_names, X_train)")
    print("report = explainer.generate_reasoning_report(X_instance, prediction_proba)")
