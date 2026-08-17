"""
Logging utilities for Credit Risk Assessment System
Provides structured logging for model training, predictions, and system events
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class CreditRiskLogger:
    """
    Custom logger for credit risk assessment system
    Provides structured logging with file and console outputs
    """
    
    def __init__(self, name: str = "CreditRisk", log_dir: Optional[str] = None):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files (default: logs/)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # Create log directory
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)s | %(message)s'
        )
        
        # File handler - detailed logs
        log_file = log_dir / f"credit_risk_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler - simplified logs
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # Error file handler - errors only
        error_file = log_dir / f"credit_risk_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(error_handler)
    
    def get_logger(self) -> logging.Logger:
        """Return the logger instance"""
        return self.logger
    
    def log_model_training(self, model_name: str, params: dict, metrics: dict):
        """
        Log model training information
        
        Args:
            model_name: Name of the model
            params: Model parameters
            metrics: Training metrics
        """
        self.logger.info(f"=== MODEL TRAINING START: {model_name} ===")
        self.logger.info(f"Parameters: {json.dumps(params, indent=2)}")
        self.logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        self.logger.info(f"=== MODEL TRAINING END: {model_name} ===")
    
    def log_prediction(self, input_data: dict, prediction: float, decision: str):
        """
        Log individual prediction
        
        Args:
            input_data: Input features (subset for privacy)
            prediction: Predicted probability
            decision: Final decision (APPROVE/REJECT)
        """
        self.logger.info(f"PREDICTION | Decision: {decision} | Probability: {prediction:.4f}")
        self.logger.debug(f"Input data: {json.dumps(input_data, indent=2)}")
    
    def log_data_processing(self, stage: str, input_shape: tuple, output_shape: tuple):
        """
        Log data processing steps
        
        Args:
            stage: Processing stage name
            input_shape: Input data shape
            output_shape: Output data shape
        """
        self.logger.info(f"DATA PROCESSING | {stage} | Input: {input_shape} → Output: {output_shape}")
    
    def log_error(self, error_type: str, error_message: str, stack_trace: Optional[str] = None):
        """
        Log errors with context
        
        Args:
            error_type: Type of error
            error_message: Error message
            stack_trace: Optional stack trace
        """
        self.logger.error(f"ERROR | {error_type} | {error_message}")
        if stack_trace:
            self.logger.error(f"Stack trace:\n{stack_trace}")
    
    def log_system_event(self, event: str, details: Optional[dict] = None):
        """
        Log system events
        
        Args:
            event: Event description
            details: Additional details
        """
        msg = f"SYSTEM EVENT | {event}"
        if details:
            msg += f" | {json.dumps(details)}"
        self.logger.info(msg)


def get_logger(name: str = "CreditRisk") -> logging.Logger:
    """
    Convenience function to get a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    credit_logger = CreditRiskLogger(name)
    return credit_logger.get_logger()


# Create default logger
default_logger = get_logger()


if __name__ == "__main__":
    # Test logging
    logger = get_logger("TestLogger")
    
    logger.info("Testing logger functionality")
    logger.debug("Debug message - detailed information")
    logger.warning("Warning message - potential issue")
    logger.error("Error message - something went wrong")
    
    # Test structured logging
    credit_logger = CreditRiskLogger("ModelTraining")
    credit_logger.log_model_training(
        model_name="XGBoost",
        params={"n_estimators": 200, "max_depth": 6},
        metrics={"accuracy": 0.85, "roc_auc": 0.87}
    )
    
    print("\n✅ Logger test complete. Check logs/ directory for output files.")
