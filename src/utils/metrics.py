"""
Evaluation Metrics for Credit Risk Assessment
Week 5: Performance Evaluation using AUC-ROC, F1-Score, Precision-Recall
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import json


class CreditRiskMetrics:
    """
    Comprehensive metrics evaluation for credit risk models
    Week 5 Requirements: AUC-ROC, F1-Score, Precision-Recall, Optimal Threshold
    """
    
    def __init__(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                 threshold: float = 0.5):
        """
        Initialize metrics calculator
        
        Args:
            y_true: True labels (0=repay, 1=default)
            y_pred_proba: Predicted probabilities for positive class
            threshold: Decision threshold
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.threshold = threshold
        self.y_pred = (y_pred_proba >= threshold).astype(int)
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        Week 5 Task: Calculate AUC-ROC, Recall, Precision
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, self.y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_true, self.y_pred_proba),
            'average_precision': average_precision_score(self.y_true, self.y_pred_proba),
            'threshold': self.threshold
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Returns:
            2x2 confusion matrix
        """
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_classification_report(self, as_dict: bool = False):
        """
        Get detailed classification report
        
        Args:
            as_dict: Return as dictionary instead of string
            
        Returns:
            Classification report
        """
        return classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=['Repay (0)', 'Default (1)'],
            output_dict=as_dict
        )
    
    def find_optimal_threshold(self, metric: str = 'f1') -> Tuple[float, float]:
        """
        Find optimal decision threshold
        Week 5 Task: Determine optimal decision threshold
        
        Args:
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        thresholds = np.arange(0.0, 1.01, 0.01)
        scores = []
        
        for thresh in thresholds:
            y_pred_temp = (self.y_pred_proba >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(self.y_true, y_pred_temp, zero_division=0)
            elif metric == 'precision':
                score = precision_score(self.y_true, y_pred_temp, zero_division=0)
            elif metric == 'recall':
                score = recall_score(self.y_true, y_pred_temp, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(self.y_true, y_pred_temp)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        optimal_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        return optimal_threshold, best_score
    
    def plot_roc_curve(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot ROC curve
        Week 5 Task: Plot the ROC curve
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=20)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None, 
                                    figsize: Tuple[int, int] = (10, 6)):
        """
        Plot Precision-Recall curve
        Week 5 Task: Plot Precision-Recall curve
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, pad=20)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        return plt.gcf()
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix heatmap
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Repay', 'Predicted Default'],
                   yticklabels=['Actual Repay', 'Actual Default'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def plot_threshold_analysis(self, save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 6)):
        """
        Plot how metrics change with different thresholds
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        thresholds = np.arange(0.0, 1.01, 0.01)
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        
        for thresh in thresholds:
            y_pred_temp = (self.y_pred_proba >= thresh).astype(int)
            precisions.append(precision_score(self.y_true, y_pred_temp, zero_division=0))
            recalls.append(recall_score(self.y_true, y_pred_temp, zero_division=0))
            f1_scores.append(f1_score(self.y_true, y_pred_temp, zero_division=0))
            accuracies.append(accuracy_score(self.y_true, y_pred_temp))
        
        plt.figure(figsize=figsize)
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
        plt.axvline(x=self.threshold, color='red', linestyle='--', 
                   label=f'Current Threshold ({self.threshold})')
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Metrics vs Decision Threshold', fontsize=14, pad=20)
        plt.legend(loc='best', fontsize=11)
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold analysis saved to {save_path}")
        
        return plt.gcf()
    
    def generate_evaluation_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            save_path: Path to save report as text file
            
        Returns:
            Report string
        """
        metrics = self.calculate_all_metrics()
        cm = self.get_confusion_matrix()
        optimal_f1_thresh, optimal_f1_score = self.find_optimal_threshold('f1')
        
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║         CREDIT RISK MODEL EVALUATION REPORT                        ║
╚════════════════════════════════════════════════════════════════════╝

PERFORMANCE METRICS (at threshold = {self.threshold:.2f})
────────────────────────────────────────────────────────────────────
• Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
• Precision:          {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
• Recall:             {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
• F1-Score:           {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)
• ROC-AUC:            {metrics['roc_auc']:.4f} ⭐ (Industry Standard)
• Average Precision:  {metrics['average_precision']:.4f}

CONFUSION MATRIX
────────────────────────────────────────────────────────────────────
                    Predicted Repay    Predicted Default
Actual Repay        {cm[0, 0]:>15d}    {cm[0, 1]:>17d}
Actual Default      {cm[1, 0]:>15d}    {cm[1, 1]:>17d}

True Negatives (TN):  {cm[0, 0]:,} - Correctly predicted repayments
False Positives (FP): {cm[0, 1]:,} - Incorrectly predicted defaults
False Negatives (FN): {cm[1, 0]:,} - Missed defaults (Type II error)
True Positives (TP):  {cm[1, 1]:,} - Correctly predicted defaults

OPTIMAL THRESHOLD ANALYSIS
────────────────────────────────────────────────────────────────────
• Optimal F1 Threshold: {optimal_f1_thresh:.2f}
• F1-Score at Optimal:  {optimal_f1_score:.4f}
• Current Threshold:    {self.threshold:.2f}

BUSINESS IMPACT ANALYSIS
────────────────────────────────────────────────────────────────────
• False Positive Rate:  {cm[0, 1] / (cm[0, 0] + cm[0, 1]):.2%}
  (Good applicants rejected - lost revenue)
  
• False Negative Rate:  {cm[1, 0] / (cm[1, 0] + cm[1, 1]):.2%}
  (Bad applicants approved - potential losses)
  
• True Positive Rate:   {cm[1, 1] / (cm[1, 0] + cm[1, 1]):.2%}
  (Correctly caught risky applicants)

INTERPRETATION
────────────────────────────────────────────────────────────────────
The model achieves an ROC-AUC of {metrics['roc_auc']:.3f}, which is the industry
standard metric for credit risk assessment. This indicates {'excellent' if metrics['roc_auc'] > 0.85 else 'good' if metrics['roc_auc'] > 0.75 else 'moderate'} 
discriminatory power between defaulters and non-defaulters.

Recall of {metrics['recall']:.2%} means the model catches {metrics['recall']*100:.1f}% of actual
defaulters, while precision of {metrics['precision']:.2%} means {metrics['precision']*100:.1f}% of
predicted defaults are correct.

════════════════════════════════════════════════════════════════════
Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
════════════════════════════════════════════════════════════════════
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to {save_path}")
        
        return report
    
    def save_metrics_json(self, save_path: str):
        """
        Save metrics as JSON for programmatic access
        
        Args:
            save_path: Path to save JSON file
        """
        metrics = self.calculate_all_metrics()
        metrics['confusion_matrix'] = self.get_confusion_matrix().tolist()
        metrics['classification_report'] = self.get_classification_report(as_dict=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics JSON saved to {save_path}")


def evaluate_model(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                  threshold: float = 0.5, save_dir: Optional[str] = None) -> CreditRiskMetrics:
    """
    Convenience function for comprehensive model evaluation
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Decision threshold
        save_dir: Directory to save all outputs
        
    Returns:
        CreditRiskMetrics instance
    """
    evaluator = CreditRiskMetrics(y_true, y_pred_proba, threshold)
    
    print("Calculating metrics...")
    metrics = evaluator.calculate_all_metrics()
    print("\nMetrics Summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    if save_dir:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving evaluation outputs to {save_dir}...")
        evaluator.plot_roc_curve(save_path=save_dir / "roc_curve.png")
        evaluator.plot_precision_recall_curve(save_path=save_dir / "precision_recall_curve.png")
        evaluator.plot_confusion_matrix(save_path=save_dir / "confusion_matrix.png")
        evaluator.plot_threshold_analysis(save_path=save_dir / "threshold_analysis.png")
        evaluator.generate_evaluation_report(save_path=save_dir / "evaluation_report.txt")
        evaluator.save_metrics_json(save_path=save_dir / "metrics.json")
        
        print("✅ All evaluation outputs saved!")
    
    return evaluator


if __name__ == "__main__":
    print("Credit Risk Metrics Module")
    print("==========================")
    print("\nFeatures:")
    print("- Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC")
    print("- ROC curve and Precision-Recall curve plotting")
    print("- Confusion matrix visualization")
    print("- Optimal threshold analysis")
    print("- Detailed evaluation reports")
