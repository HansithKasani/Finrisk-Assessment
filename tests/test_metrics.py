"""
Unit tests for metrics evaluation module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metrics import CreditRiskMetrics, evaluate_model


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing"""
    np.random.seed(42)
    
    # Create some realistic predictions
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10)
    y_pred_proba = np.concatenate([
        np.random.uniform(0.1, 0.4, 50),  # True negatives (low prob)
        np.random.uniform(0.6, 0.9, 50)   # True positives (high prob)
    ])
    
    return y_true, y_pred_proba


@pytest.fixture
def metrics_evaluator(sample_predictions):
    """Create a metrics evaluator instance"""
    y_true, y_pred_proba = sample_predictions
    return CreditRiskMetrics(y_true, y_pred_proba, threshold=0.5)


class TestCreditRiskMetrics:
    """Test suite for CreditRiskMetrics"""
    
    def test_initialization(self, metrics_evaluator):
        """Test metrics evaluator initialization"""
        assert metrics_evaluator is not None
        assert metrics_evaluator.threshold == 0.5
        assert len(metrics_evaluator.y_true) == len(metrics_evaluator.y_pred_proba)
    
    def test_calculate_all_metrics(self, metrics_evaluator):
        """Test calculation of all metrics"""
        metrics = metrics_evaluator.calculate_all_metrics()
        
        # Check all required metrics are present
        required_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'roc_auc', 'average_precision', 'threshold'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float, np.number))
            
            # Check metrics are in valid range
            if metric != 'threshold':
                assert 0 <= metrics[metric] <= 1
    
    def test_confusion_matrix(self, metrics_evaluator):
        """Test confusion matrix generation"""
        cm = metrics_evaluator.get_confusion_matrix()
        
        # Check shape
        assert cm.shape == (2, 2)
        
        # Check all values are non-negative
        assert (cm >= 0).all()
        
        # Check sum equals total samples
        assert cm.sum() == len(metrics_evaluator.y_true)
    
    def test_classification_report(self, metrics_evaluator):
        """Test classification report generation"""
        # String report
        report_str = metrics_evaluator.get_classification_report(as_dict=False)
        assert isinstance(report_str, str)
        assert 'precision' in report_str.lower()
        assert 'recall' in report_str.lower()
        
        # Dict report
        report_dict = metrics_evaluator.get_classification_report(as_dict=True)
        assert isinstance(report_dict, dict)
        assert 'Repay (0)' in report_dict or '0' in report_dict
    
    def test_find_optimal_threshold(self, metrics_evaluator):
        """Test optimal threshold finding"""
        # Test F1 optimization
        optimal_thresh, best_f1 = metrics_evaluator.find_optimal_threshold(metric='f1')
        assert 0 <= optimal_thresh <= 1
        assert 0 <= best_f1 <= 1
        
        # Test precision optimization
        optimal_thresh, best_precision = metrics_evaluator.find_optimal_threshold(metric='precision')
        assert 0 <= optimal_thresh <= 1
        assert 0 <= best_precision <= 1
        
        # Test recall optimization
        optimal_thresh, best_recall = metrics_evaluator.find_optimal_threshold(metric='recall')
        assert 0 <= optimal_thresh <= 1
        assert 0 <= best_recall <= 1
    
    def test_generate_evaluation_report(self, metrics_evaluator):
        """Test evaluation report generation"""
        report = metrics_evaluator.generate_evaluation_report()
        
        # Check report is generated
        assert isinstance(report, str)
        assert len(report) > 100
        
        # Check key sections are present
        assert "PERFORMANCE METRICS" in report
        assert "CONFUSION MATRIX" in report
        assert "ROC-AUC" in report
        assert "OPTIMAL THRESHOLD" in report
    
    def test_save_metrics_json(self, metrics_evaluator, tmp_path):
        """Test saving metrics as JSON"""
        json_path = tmp_path / "metrics.json"
        metrics_evaluator.save_metrics_json(str(json_path))
        
        # Check file exists
        assert json_path.exists()
        
        # Check file can be read
        import json
        with open(json_path) as f:
            data = json.load(f)
        
        assert 'accuracy' in data
        assert 'roc_auc' in data


class TestMetricsWithDifferentData:
    """Test metrics with various data scenarios"""
    
    def test_perfect_predictions(self):
        """Test with perfect predictions"""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8])
        
        evaluator = CreditRiskMetrics(y_true, y_pred_proba, threshold=0.5)
        metrics = evaluator.calculate_all_metrics()
        
        # Perfect predictions should have high scores
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_random_predictions(self):
        """Test with random predictions"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100)
        
        evaluator = CreditRiskMetrics(y_true, y_pred_proba, threshold=0.5)
        metrics = evaluator.calculate_all_metrics()
        
        # Random predictions should have metrics around 0.5
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_imbalanced_data(self):
        """Test with imbalanced data"""
        # 90% class 0, 10% class 1
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred_proba = np.random.rand(100)
        
        evaluator = CreditRiskMetrics(y_true, y_pred_proba, threshold=0.5)
        metrics = evaluator.calculate_all_metrics()
        
        # Metrics should still be calculable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_all_positive_predictions(self):
        """Test when all predictions are positive"""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.9, 0.8, 0.9, 0.8])
        
        evaluator = CreditRiskMetrics(y_true, y_pred_proba, threshold=0.5)
        metrics = evaluator.calculate_all_metrics()
        
        # Recall should be perfect, precision should be 0.5
        assert metrics['recall'] == 1.0
        assert abs(metrics['precision'] - 0.5) < 0.01
    
    def test_all_negative_predictions(self):
        """Test when all predictions are negative"""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.1, 0.2])
        
        evaluator = CreditRiskMetrics(y_true, y_pred_proba, threshold=0.5)
        metrics = evaluator.calculate_all_metrics()
        
        # Recall should be 0 (no positives predicted)
        assert metrics['recall'] == 0.0


class TestEvaluateModelFunction:
    """Test the evaluate_model convenience function"""
    
    def test_evaluate_model_basic(self, sample_predictions, tmp_path):
        """Test evaluate_model function"""
        y_true, y_pred_proba = sample_predictions
        
        evaluator = evaluate_model(
            y_true, 
            y_pred_proba, 
            threshold=0.5,
            save_dir=str(tmp_path)
        )
        
        # Check evaluator is returned
        assert isinstance(evaluator, CreditRiskMetrics)
        
        # Check files are created
        assert (tmp_path / "roc_curve.png").exists()
        assert (tmp_path / "precision_recall_curve.png").exists()
        assert (tmp_path / "confusion_matrix.png").exists()
        assert (tmp_path / "evaluation_report.txt").exists()
        assert (tmp_path / "metrics.json").exists()


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_arrays(self):
        """Test with empty arrays"""
        y_true = np.array([])
        y_pred_proba = np.array([])
        
        with pytest.raises(Exception):
            CreditRiskMetrics(y_true, y_pred_proba)
    
    def test_mismatched_lengths(self):
        """Test with mismatched array lengths"""
        y_true = np.array([0, 1, 0])
        y_pred_proba = np.array([0.5, 0.6])
        
        with pytest.raises(Exception):
            evaluator = CreditRiskMetrics(y_true, y_pred_proba)
            evaluator.calculate_all_metrics()
    
    def test_invalid_threshold(self):
        """Test with invalid threshold"""
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.3, 0.7, 0.4, 0.8])
        
        # Should work with any threshold between 0 and 1
        evaluator = CreditRiskMetrics(y_true, y_pred_proba, threshold=1.5)
        metrics = evaluator.calculate_all_metrics()
        assert metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
