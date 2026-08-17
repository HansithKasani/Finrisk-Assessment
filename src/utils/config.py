"""
Configuration management for Credit Risk Assessment System
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager"""
    
    def __init__(self, config_file: str = "config/config.yaml"):
        self.config_file = Path(config_file)
        self.config = {}
        self.env_vars = {}
        
        self._load_config()
        self._load_env_vars()
    
    def _load_config(self):
        """Load YAML configuration file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self.config = {}
        else:
            logger.warning(f"Config file not found: {self.config_file}")
    
    def _load_env_vars(self):
        """Load environment variables"""
        # Load from .env file if exists
        env_file = Path("config/.env")
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            self.env_vars[key.strip()] = value.strip()
        
        # Override with actual environment variables
        self.env_vars.update(os.environ)
    
    def get(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """Get configuration value"""
        # Check environment variables first
        if key in self.env_vars:
            return self.env_vars[key]
        
        # Then check config file
        if section:
            if section in self.config and key in self.config[section]:
                return self.config[section][key]
        else:
            if key in self.config:
                return self.config[key]
        
        # Return default
        return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {})
    
    def set(self, key: str, value: Any, section: Optional[str] = None):
        """Set configuration value"""
        if section:
            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = value
        else:
            self.config[key] = value
        
        logger.info(f"Set {section}.{key} = {value}" if section else f"Set {key} = {value}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.get_section("model")
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.get_section("data")
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature configuration"""
        return self.get_section("features")
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.get_section("training")
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.get_section("performance")
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration"""
        return self.get_section("deployment")
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_sections = ["model", "data", "training"]
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required section: {section}")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return self.config
    
    def save(self, path: str = None):
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_file
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {save_path}")


class ModelConfig:
    """Model-specific configuration"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model_config = self.config.get_model_config()
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters"""
        return self.model_config.get("hyperparameters", {})
    
    def get_n_estimators(self) -> int:
        """Get number of boosting stages"""
        return self.get_hyperparameters().get("n_estimators", 200)
    
    def get_max_depth(self) -> int:
        """Get maximum tree depth"""
        return self.get_hyperparameters().get("max_depth", 6)
    
    def get_learning_rate(self) -> float:
        """Get learning rate"""
        return self.get_hyperparameters().get("learning_rate", 0.1)
    
    def get_threshold(self) -> float:
        """Get prediction threshold"""
        return self.model_config.get("threshold", {}).get("optimized", 0.20)


class DataConfig:
    """Data-specific configuration"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_config = self.config.get_data_config()
    
    def get_raw_path(self) -> str:
        """Get raw data path"""
        return self.data_config.get("raw_path", "Data/raw/")
    
    def get_processed_path(self) -> str:
        """Get processed data path"""
        return self.data_config.get("processed_path", "Data/processed/")
    
    def get_train_test_split(self) -> float:
        """Get train-test split ratio"""
        return self.data_config.get("train_test_split", 0.8)
    
    def get_random_state(self) -> int:
        """Get random state for reproducibility"""
        return self.data_config.get("random_state", 42)
    
    def should_stratify(self) -> bool:
        """Check if stratification should be used"""
        return self.data_config.get("stratify", True)
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration"""
        return self.data_config.get("preprocessing", {})
    
    def should_handle_missing(self) -> bool:
        """Check if missing values should be handled"""
        return "handle_missing" in self.get_preprocessing_config()
    
    def get_missing_strategy(self) -> str:
        """Get missing value imputation strategy"""
        return self.get_preprocessing_config().get("handle_missing", "median")


class FeatureConfig:
    """Feature-specific configuration"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.feature_config = self.config.get_feature_config()
    
    def get_total_features(self) -> int:
        """Get total number of features"""
        return self.feature_config.get("total_count", 85)
    
    def get_feature_categories(self) -> Dict[str, Any]:
        """Get feature categories"""
        return self.feature_config.get("categories", {})
    
    def get_demographics_features(self) -> int:
        """Get number of demographic features"""
        return self.get_feature_categories().get("demographics", {}).get("count", 9)
    
    def get_financial_features(self) -> int:
        """Get number of financial features"""
        return self.get_feature_categories().get("financial", {}).get("count", 15)
    
    def get_credit_bureau_features(self) -> int:
        """Get number of credit bureau features"""
        return self.get_feature_categories().get("credit_bureau", {}).get("count", 6)
    
    def get_engineered_features(self) -> int:
        """Get number of engineered features"""
        return self.get_feature_categories().get("engineered", {}).get("count", 3)
    
    def get_all_feature_names(self) -> list:
        """Get all feature names"""
        features = []
        for category in self.get_feature_categories().values():
            features.extend(category.get("features", []))
        return features


class PerformanceConfig:
    """Performance monitoring configuration"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.perf_config = self.config.get_performance_config()
    
    def get_target_metrics(self) -> Dict[str, float]:
        """Get target performance metrics"""
        return self.perf_config.get("target_metrics", {})
    
    def get_min_accuracy(self) -> float:
        """Get minimum acceptable accuracy"""
        return self.get_target_metrics().get("accuracy", 0.85)
    
    def get_min_precision(self) -> float:
        """Get minimum acceptable precision"""
        return self.get_target_metrics().get("precision", 0.80)
    
    def get_min_recall(self) -> float:
        """Get minimum acceptable recall"""
        return self.get_target_metrics().get("recall", 0.75)
    
    def get_min_roc_auc(self) -> float:
        """Get minimum acceptable ROC-AUC"""
        return self.get_target_metrics().get("roc_auc", 0.85)
    
    def get_degradation_threshold(self) -> float:
        """Get performance degradation threshold for retraining"""
        monitoring = self.perf_config.get("monitoring", {})
        return monitoring.get("performance_degradation_threshold", 0.80)


# Create global config instance
_config_instance = None


def get_config() -> Config:
    """Get global config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reset_config():
    """Reset global config instance"""
    global _config_instance
    _config_instance = None
