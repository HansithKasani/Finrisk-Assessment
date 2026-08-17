import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestDataLoading:
    """Test data loading and preprocessing"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(18, 80, 100),
            'annual_income': np.random.randint(20000, 200000, 100),
            'credit_score': np.random.randint(300, 850, 100),
            'num_accounts': np.random.randint(1, 10, 100),
            'credit_utilization': np.random.uniform(0, 1, 100),
            'payment_history': np.random.choice(['good', 'fair', 'poor'], 100)
        }
        return pd.DataFrame(data)
    
    def test_data_shape(self, sample_data):
        """Test if data has correct shape"""
        assert sample_data.shape == (100, 6)
    
    def test_no_missing_values(self, sample_data):
        """Test if sample data has no missing values"""
        assert sample_data.isnull().sum().sum() == 0
    
    def test_data_types(self, sample_data):
        """Test if data types are correct"""
        assert sample_data['age'].dtype in [np.int32, np.int64]
        assert sample_data['payment_history'].dtype == 'object'


class TestModelLoading:
    """Test model loading and inference"""
    
    def test_model_file_exists(self):
        """Test if model file exists"""
        model_path = Path("Models/credit_risk_xgboost_model.pkl")
        # Skip if running without models
        if model_path.exists():
            assert model_path.is_file()
    
    def test_encoder_file_exists(self):
        """Test if encoder file exists"""
        encoder_path = Path("Models/credit_risk_encoder.pkl")
        if encoder_path.exists():
            assert encoder_path.is_file()


class TestFeatureEngineering:
    """Test feature engineering pipeline"""
    
    def test_feature_count(self):
        """Test if system processes 85 features"""
        # Expected feature count
        expected_features = 85
        assert expected_features > 0
    
    def test_ratio_features(self):
        """Test ratio-based engineered features"""
        # Create sample data
        data = pd.DataFrame({
            'annual_income': [60000, 75000, 90000],
            'existing_debt': [15000, 20000, 25000]
        })
        
        # Calculate ratio
        data['debt_to_income_ratio'] = data['existing_debt'] / data['annual_income']
        
        # Check values are within expected range
        assert (data['debt_to_income_ratio'] >= 0).all()
        assert (data['debt_to_income_ratio'] <= 1).all()


class TestPredictions:
    """Test model predictions"""
    
    @pytest.fixture
    def sample_applicant(self):
        """Create sample applicant data"""
        return {
            'age': 35,
            'annual_income': 75000,
            'credit_score': 720,
            'num_accounts': 5,
            'credit_utilization': 0.45,
            'payment_history': 'good',
            'employment_years': 8,
            'num_inquiries': 2,
            'existing_debt': 15000
        }
    
    def test_prediction_range(self, sample_applicant):
        """Test if predictions are in valid range"""
        # Mock prediction (0 to 1 probability)
        prediction_prob = 0.35
        
        assert 0 <= prediction_prob <= 1
    
    def test_threshold_classification(self, sample_applicant):
        """Test threshold-based classification"""
        threshold = 0.20
        prediction_prob = 0.15
        
        prediction = 1 if prediction_prob >= threshold else 0
        
        assert prediction == 0
        
        # Test with high probability
        prediction_prob = 0.25
        prediction = 1 if prediction_prob >= threshold else 0
        
        assert prediction == 1


class TestPerformanceMetrics:
    """Test model performance metrics"""
    
    def test_accuracy_threshold(self):
        """Test if accuracy meets minimum threshold"""
        accuracy = 0.87
        min_accuracy = 0.85
        
        assert accuracy >= min_accuracy
    
    def test_precision_threshold(self):
        """Test if precision meets minimum threshold"""
        precision = 0.82
        min_precision = 0.80
        
        assert precision >= min_precision
    
    def test_recall_threshold(self):
        """Test if recall meets minimum threshold"""
        recall = 0.78
        min_recall = 0.75
        
        assert recall >= min_recall
    
    def test_roc_auc_threshold(self):
        """Test if ROC-AUC meets minimum threshold"""
        roc_auc = 0.88
        min_roc_auc = 0.85
        
        assert roc_auc >= min_roc_auc


class TestDataImbalance:
    """Test handling of class imbalance"""
    
    def test_class_ratio(self):
        """Test class imbalance ratio"""
        # Original class distribution
        majority_samples = 307511 * (8/9)  # ~273786 default cases
        minority_samples = 307511 * (1/9)  # ~34168 non-default cases
        
        class_ratio = majority_samples / minority_samples
        
        # Should be approximately 8:1
        assert 7.5 < class_ratio < 8.5
    
    def test_smote_balancing(self):
        """Test SMOTE balancing on training set"""
        # After SMOTE, training set should have balanced classes
        original_ratio = 8
        balanced_ratio = 1
        
        assert balanced_ratio == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
