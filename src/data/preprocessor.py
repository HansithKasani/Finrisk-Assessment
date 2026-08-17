"""
Data Preprocessing Module for Credit Risk Assessment
Handles missing values, encoding, scaling, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for credit risk data.
    Implements Week 2 requirements: cleaning, encoding, engineering, scaling.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        
    def identify_feature_types(self, df):
        """Automatically identify numeric and categorical features"""
        self.numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target if present
        if 'TARGET' in self.numeric_features:
            self.numeric_features.remove('TARGET')
        if 'SK_ID_CURR' in self.numeric_features:
            self.numeric_features.remove('SK_ID_CURR')
            
        logger.info(f"Identified {len(self.numeric_features)} numeric features")
        logger.info(f"Identified {len(self.categorical_features)} categorical features")
        
    def handle_missing_values(self, df, fit=True):
        """
        Handle missing values using median for numeric and mode for categorical
        Week 2 Task: Median imputation for numerical, mode for categorical
        """
        df_copy = df.copy()
        
        if fit:
            # Fit and transform numeric features
            if self.numeric_features:
                df_copy[self.numeric_features] = self.numeric_imputer.fit_transform(
                    df_copy[self.numeric_features]
                )
            
            # Fit and transform categorical features
            if self.categorical_features:
                df_copy[self.categorical_features] = self.categorical_imputer.fit_transform(
                    df_copy[self.categorical_features].astype(str)
                )
        else:
            # Only transform
            if self.numeric_features:
                df_copy[self.numeric_features] = self.numeric_imputer.transform(
                    df_copy[self.numeric_features]
                )
            
            if self.categorical_features:
                df_copy[self.categorical_features] = self.categorical_imputer.transform(
                    df_copy[self.categorical_features].astype(str)
                )
        
        logger.info("Missing values handled successfully")
        return df_copy
    
    def encode_categorical_features(self, df, fit=True):
        """
        Apply LabelEncoder to categorical features
        Week 2 Task: Apply LabelEncoder or OneHotEncoder to categorical features
        """
        df_copy = df.copy()
        
        for col in self.categorical_features:
            if col in df_copy.columns:
                if fit:
                    # Fit and transform
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # Only transform with fitted encoder
                    if col in self.label_encoders:
                        # Handle unseen labels
                        le = self.label_encoders[col]
                        df_copy[col] = df_copy[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        logger.info(f"Encoded {len(self.categorical_features)} categorical features")
        return df_copy
    
    def create_engineered_features(self, df):
        """
        Create new features like Debt_to_Income_Ratio and Credit_to_Annuity_Ratio
        Week 2 Task: Create features like Debt_to_Income_Ratio and Credit_to_Annuity_Ratio
        """
        df_copy = df.copy()
        
        # Debt to Income Ratio
        if 'AMT_CREDIT' in df_copy.columns and 'AMT_INCOME_TOTAL' in df_copy.columns:
            df_copy['DEBT_TO_INCOME_RATIO'] = df_copy['AMT_CREDIT'] / (df_copy['AMT_INCOME_TOTAL'] + 1)
        
        # Credit to Annuity Ratio
        if 'AMT_CREDIT' in df_copy.columns and 'AMT_ANNUITY' in df_copy.columns:
            df_copy['CREDIT_TO_ANNUITY_RATIO'] = df_copy['AMT_CREDIT'] / (df_copy['AMT_ANNUITY'] + 1)
        
        # Income per Family Member
        if 'AMT_INCOME_TOTAL' in df_copy.columns and 'CNT_FAM_MEMBERS' in df_copy.columns:
            df_copy['INCOME_PER_PERSON'] = df_copy['AMT_INCOME_TOTAL'] / (df_copy['CNT_FAM_MEMBERS'] + 1)
        
        # Payment Rate
        if 'AMT_ANNUITY' in df_copy.columns and 'AMT_INCOME_TOTAL' in df_copy.columns:
            df_copy['PAYMENT_RATE'] = df_copy['AMT_ANNUITY'] / (df_copy['AMT_INCOME_TOTAL'] + 1)
        
        # Days employed to age ratio
        if 'DAYS_EMPLOYED' in df_copy.columns and 'DAYS_BIRTH' in df_copy.columns:
            df_copy['EMPLOYED_TO_AGE_RATIO'] = df_copy['DAYS_EMPLOYED'] / (df_copy['DAYS_BIRTH'] + 1)
        
        # Credit per child
        if 'AMT_CREDIT' in df_copy.columns and 'CNT_CHILDREN' in df_copy.columns:
            df_copy['CREDIT_PER_CHILD'] = df_copy['AMT_CREDIT'] / (df_copy['CNT_CHILDREN'] + 1)
        
        logger.info("Created 6 engineered features")
        return df_copy
    
    def scale_features(self, df, fit=True, exclude_cols=None):
        """
        Use StandardScaler to normalize numeric features
        Week 2 Task: Use StandardScaler to normalize numeric features
        """
        df_copy = df.copy()
        
        if exclude_cols is None:
            exclude_cols = ['TARGET', 'SK_ID_CURR']
        
        # Get columns to scale
        cols_to_scale = [col for col in df_copy.columns 
                        if col not in exclude_cols and 
                        df_copy[col].dtype in ['int64', 'float64']]
        
        if fit:
            df_copy[cols_to_scale] = self.scaler.fit_transform(df_copy[cols_to_scale])
        else:
            df_copy[cols_to_scale] = self.scaler.transform(df_copy[cols_to_scale])
        
        logger.info(f"Scaled {len(cols_to_scale)} features using StandardScaler")
        return df_copy
    
    def fit_transform(self, df, target_col='TARGET'):
        """
        Complete preprocessing pipeline: fit and transform
        """
        logger.info("Starting preprocessing pipeline (fit_transform)...")
        df_copy = df.copy()
        
        # Separate target
        y = None
        if target_col in df_copy.columns:
            y = df_copy[target_col]
            df_copy = df_copy.drop(columns=[target_col])
        
        # Step 1: Identify feature types
        self.identify_feature_types(df_copy)
        
        # Step 2: Handle missing values
        df_copy = self.handle_missing_values(df_copy, fit=True)
        
        # Step 3: Encode categorical features
        df_copy = self.encode_categorical_features(df_copy, fit=True)
        
        # Step 4: Create engineered features
        df_copy = self.create_engineered_features(df_copy)
        
        # Step 5: Scale numeric features
        df_copy = self.scale_features(df_copy, fit=True, 
                                     exclude_cols=['SK_ID_CURR'])
        
        # Store feature names
        self.feature_names = df_copy.columns.tolist()
        
        logger.info(f"Preprocessing complete. Final shape: {df_copy.shape}")
        
        if y is not None:
            return df_copy, y
        return df_copy
    
    def transform(self, df, target_col='TARGET'):
        """
        Transform new data using fitted preprocessor
        """
        logger.info("Starting preprocessing pipeline (transform)...")
        df_copy = df.copy()
        
        # Separate target
        y = None
        if target_col in df_copy.columns:
            y = df_copy[target_col]
            df_copy = df_copy.drop(columns=[target_col])
        
        # Apply transformations
        df_copy = self.handle_missing_values(df_copy, fit=False)
        df_copy = self.encode_categorical_features(df_copy, fit=False)
        df_copy = self.create_engineered_features(df_copy)
        df_copy = self.scale_features(df_copy, fit=False, 
                                     exclude_cols=['SK_ID_CURR'])
        
        # Ensure same columns as training
        if self.feature_names is not None:
            missing_cols = set(self.feature_names) - set(df_copy.columns)
            for col in missing_cols:
                df_copy[col] = 0
            df_copy = df_copy[self.feature_names]
        
        logger.info(f"Transform complete. Final shape: {df_copy.shape}")
        
        if y is not None:
            return df_copy, y
        return df_copy
    
    def save(self, filepath):
        """Save the fitted preprocessor"""
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load a fitted preprocessor"""
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def prepare_data_for_modeling(data_path, save_preprocessor=True):
    """
    Convenience function to prepare data for modeling
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor
    preprocessor = CreditDataPreprocessor()
    
    # Fit and transform
    X, y = preprocessor.fit_transform(df)
    
    # Save preprocessor if requested
    if save_preprocessor:
        preprocessor.save('Models/credit_risk_preprocessor.pkl')
    
    return X, y, preprocessor


if __name__ == "__main__":
    # Example usage
    print("Credit Data Preprocessor Module")
    print("================================")
    print("\nUsage:")
    print("from src.data.preprocessor import CreditDataPreprocessor")
    print("\npreprocessor = CreditDataPreprocessor()")
    print("X, y = preprocessor.fit_transform(df)")
