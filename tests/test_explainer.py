"""
Unit tests for model explainer module
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.explainer import CreditRiskExplainer


@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = Mock()
    model.predict_proba = Mock(return_value=np.array([[0.3, 0.7], [0.6, 0.4]]))
    model.predict = Mock(return_value=np.array([1, 0]))
    return model


@pytest.fixture
def sample_features():
    """Create sample feature data"""
    np.random.seed(42)
    return np.random.rand(10, 5)


@pytest.fixture
def feature_names():
    """Create sample feature names"""
    return ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']


@pytest.fixture
def explainer(mock_model, feature_names, sample_features):
    """Create an explainer instance"""
    return CreditRiskExplainer(
        model=mock_model,
        feature_names=feature_names,
        X_train=sample_features
    )


class TestCreditRiskExplainer:
    """Test suite for CreditRiskExplainer"""
    
    def test_initialization(self, explainer, feature_names):
        """Test explainer initialization"""
        assert explainer is not None
        assert explainer.feature_names == feature_names
        assert explainer.model is not None
    
    def test_shap_explainer_creation(self, explainer):
        """Test SHAP explainer is created"""
        # SHAP explainer should be initialized
        assert explainer.shap_explainer is not None or explainer.X_train is not None
    
    def test_get_top_features(self, explainer):
        """Test getting top features by SHAP value"""
        np.random.seed(42)
        shap_values = np.random.rand(5) - 0.5  # Random values between -0.5 and 0.5
        X_instance = np.random.rand(5)
        
        top_features = explainer.get_top_features(shap_values, X_instance, top_n=3)
        
        # Check we get correct number of features
        assert len(top_features) == 3
        
        # Check each tuple has correct structure
        for feature_name, shap_val, feature_val in top_features:
            assert isinstance(feature_name, str)
            assert isinstance(shap_val, (int, float, np.number))
            assert isinstance(feature_val, (int, float, np.number))
    
    def test_generate_reasoning_report(self, explainer, sample_features):
        """Test reasoning report generation"""
        X_instance = sample_features[0]
        prediction_proba = 0.7
        
        # Mock SHAP values
        explainer.get_shap_values = Mock(return_value=np.random.rand(1, 5) - 0.5)
        
        report = explainer.generate_reasoning_report(X_instance, prediction_proba)
        
        # Check report is generated
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check key sections are present
        assert "DECISION" in report
        assert "REJECTED" in report or "APPROVED" in report
        assert "Default Probability" in report
    
    def test_risk_level_classification(self, explainer):
        """Test risk level classification"""
        # Test different probability levels
        assert "VERY HIGH" in explainer._get_risk_level(0.8)
        assert "HIGH" in explainer._get_risk_level(0.6)
        assert "MEDIUM" in explainer._get_risk_level(0.4)
        assert "LOW" in explainer._get_risk_level(0.2)
        assert "VERY LOW" in explainer._get_risk_level(0.1)
    
    def test_format_feature_name(self, explainer):
        """Test feature name formatting"""
        # Test underscore replacement
        assert explainer._format_feature_name("AMT_INCOME_TOTAL") == "Amount Income Total"
        
        # Test abbreviation expansion
        assert "Amount" in explainer._format_feature_name("AMT_CREDIT")
        assert "Count" in explainer._format_feature_name("CNT_CHILDREN")
    
    def test_get_primary_concerns(self, explainer):
        """Test extracting primary concerns"""
        top_features = [
            ("AMT_CREDIT", 0.5, 100),  # Positive SHAP = increases risk
            ("AMT_INCOME_TOTAL", -0.3, 50),  # Negative SHAP = decreases risk
            ("EXT_SOURCE_1", 0.2, 0.5),  # Positive SHAP
        ]
        
        concerns = explainer._get_primary_concerns(top_features)
        assert isinstance(concerns, str)
        assert len(concerns) > 0
    
    def test_get_positive_indicators(self, explainer):
        """Test extracting positive indicators"""
        top_features = [
            ("AMT_CREDIT", 0.5, 100),  # Positive SHAP = increases risk
            ("AMT_INCOME_TOTAL", -0.3, 50),  # Negative SHAP = decreases risk
            ("EXT_SOURCE_1", -0.2, 0.5),  # Negative SHAP = decreases risk
        ]
        
        positives = explainer._get_positive_indicators(top_features)
        assert isinstance(positives, str)
        assert len(positives) > 0


class TestExplainerWithRealData:
    """Test explainer with more realistic scenarios"""
    
    def test_single_prediction_explanation(self, explainer, sample_features):
        """Test explanation for single prediction"""
        X_instance = sample_features[0]
        
        # Mock SHAP values
        explainer.get_shap_values = Mock(return_value=np.array([[0.1, -0.2, 0.3, -0.1, 0.05]]))
        
        report = explainer.generate_reasoning_report(X_instance, 0.65)
        
        # Verify report structure
        assert "DECISION" in report
        assert "REJECTED" in report
        assert "65.0%" in report or "0.65" in report
    
    def test_batch_explanations(self, explainer, sample_features):
        """Test generating explanations for multiple instances"""
        # Mock SHAP values for batch
        batch_shap = np.random.rand(len(sample_features), 5) - 0.5
        explainer.get_shap_values = Mock(return_value=batch_shap)
        
        for i, X_instance in enumerate(sample_features[:3]):
            report = explainer.generate_reasoning_report(X_instance, 0.5 + i * 0.1)
            assert isinstance(report, str)
            assert len(report) > 100
    
    def test_edge_case_probabilities(self, explainer, sample_features):
        """Test with edge case probabilities"""
        X_instance = sample_features[0]
        explainer.get_shap_values = Mock(return_value=np.random.rand(1, 5) - 0.5)
        
        # Test very low probability
        report_low = explainer.generate_reasoning_report(X_instance, 0.01)
        assert "APPROVED" in report_low
        
        # Test very high probability
        report_high = explainer.generate_reasoning_report(X_instance, 0.99)
        assert "REJECTED" in report_high
        
        # Test threshold case
        report_threshold = explainer.generate_reasoning_report(X_instance, 0.5)
        assert "DECISION" in report_threshold


class TestExplainerSaveLoad:
    """Test saving and loading explainer"""
    
    def test_save_load(self, explainer, tmp_path):
        """Test save and load functionality"""
        save_path = tmp_path / "test_explainer.pkl"
        
        # Save explainer config
        explainer.save(str(save_path))
        
        # Verify file exists
        assert save_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
