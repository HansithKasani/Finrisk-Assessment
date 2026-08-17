"""
Data loading and preprocessing utilities for Credit Risk Assessment System
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage credit risk data"""
    
    def __init__(self, data_path: str = "Data/raw/"):
        self.data_path = Path(data_path)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self, filename: str = "application_train_cleaned.csv") -> pd.DataFrame:
        """Load CSV data file"""
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        self.data = pd.read_csv(file_path)
        logger.info(f"Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
        
        return self.data
    
    def get_data_info(self) -> dict:
        """Get basic data information"""
        if self.data is None:
            return {}
        
        return {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
            "dtypes": self.data.dtypes.to_dict()
        }
    
    def handle_missing_values(self, strategy: str = "median"):
        """Handle missing values"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                if self.data[col].dtype in ['float64', 'int64']:
                    if strategy == "median":
                        self.data[col].fillna(self.data[col].median(), inplace=True)
                    elif strategy == "mean":
                        self.data[col].fillna(self.data[col].mean(), inplace=True)
                else:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        
        logger.info("Missing values handled")
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42, 
                   target_col: str = "TARGET", stratify: bool = True):
        """Split data into training and test sets"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info(f"Splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
        
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        
        stratify_y = y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y
        )
        
        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_feature_types(self):
        """Identify numerical and categorical features"""
        numerical = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical = self.data.select_dtypes(include=['object']).columns.tolist()
        
        return {
            "numerical": numerical,
            "categorical": categorical
        }
    
    def get_class_balance(self, target_col: str = "TARGET"):
        """Get class distribution"""
        if self.data is None:
            return {}
        
        value_counts = self.data[target_col].value_counts()
        total = len(self.data)
        
        return {
            col: {"count": count, "percentage": (count/total)*100}
            for col, count in value_counts.items()
        }


class DataPreprocessor:
    """Preprocess features"""
    
    def __init__(self):
        self.encoder = None
        self.scaler = None
        self.categorical_features = None
        self.numerical_features = None
    
    def encode_categorical(self, X, categorical_cols=None, fit=True):
        """Encode categorical features"""
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        self.categorical_features = categorical_cols
        
        if fit:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            X_encoded = X.copy()
            X_encoded[categorical_cols] = self.encoder.fit_transform(X[categorical_cols])
        else:
            X_encoded = X.copy()
            X_encoded[categorical_cols] = self.encoder.transform(X[categorical_cols])
        
        return X_encoded
    
    def scale_numerical(self, X, numerical_cols=None, fit=True):
        """Scale numerical features"""
        if numerical_cols is None:
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.numerical_features = numerical_cols
        
        if fit:
            self.scaler = StandardScaler()
            X_scaled = X.copy()
            X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X_scaled = X.copy()
            X_scaled[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X_scaled
    
    def preprocess(self, X, fit=True):
        """Full preprocessing pipeline"""
        X_processed = self.encode_categorical(X, fit=fit)
        X_processed = self.scale_numerical(X_processed, fit=fit)
        
        return X_processed
