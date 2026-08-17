"""
Unit tests for data preprocessing module
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import CreditDataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample credit risk data for testing"""
    np.random.seed(42)
    
    data = {
        'TARGET': [0, 1, 0, 1, 0],
        'SK_ID_CURR': [100001, 100002, 100003, 100004, 100005],
        'CODE_GENDER': ['M', 'F', 'M', 'F', 'M'],
        'FLAG_OWN_CAR': ['Y', 'N', 'Y', 'N', 'Y'],
        'CNT_CHILDREN': [0, 1, 2, 0, 1],
        'AMT_INCOME_TOTAL': [150000, 120000, 200000, 180000, 160000],
        'AMT_CREDIT': [500000, 400000, 600000, 550000, 480000],
        'AMT_ANNUITY': [25000, 20000, 30000, 27500, 24000],
        'DAYS_BIRTH': [-14000, -12000, -16000, -15000, -13000],
        'DAYS_EMPLOYED': [-2000, -1500, -3000, -2500, -1800],
        'EXT_SOURCE_1': [0.5, 0.6, 0.7, 0.4, 0.55],
        'EXT_SOURCE_2': [0.6, 0.5, 0.8, 0.45, 0.6],
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance"""
    return CreditDataPreprocessor()


class TestCreditDataPreprocessor:
    """Test suite for CreditDataPreprocessor"""
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor is not None
        assert preprocessor.label_encoders == {}
        assert preprocessor.scaler is not None
    
    def test_identify_feature_types(self, preprocessor, sample_data):
        """Test feature type identification"""
        df = sample_data.copy()
        preprocessor.identify_feature_types(df)
        
        assert preprocessor.numeric_features is not None
        assert preprocessor.categorical_features is not None
        assert 'CODE_GENDER' in preprocessor.categorical_features
        assert 'AMT_INCOME_TOTAL' in preprocessor.numeric_features
        assert 'TARGET' not in preprocessor.numeric_features
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling"""
        df = sample_data.copy()
        
        # Add some missing values
        df.loc[0, 'AMT_INCOME_TOTAL'] = np.nan
        df.loc[1, 'CODE_GENDER'] = np.nan
        
        preprocessor.identify_feature_types(df)
        df_clean = preprocessor.handle_missing_values(df, fit=True)
        
        # Check no missing values remain
        assert df_clean.isnull().sum().sum() == 0
    
    def test_encode_categorical_features(self, preprocessor, sample_data):
        """Test categorical encoding"""
        df = sample_data.copy()
        
        preprocessor.identify_feature_types(df)
        df_encoded = preprocessor.encode_categorical_features(df, fit=True)
        
        # Check that categorical features are now numeric
        for col in preprocessor.categorical_features:
            if col in df_encoded.columns:
                assert pd.api.types.is_numeric_dtype(df_encoded[col])
        
        # Check encoders were created
        assert len(preprocessor.label_encoders) > 0
    
    def test_create_engineered_features(self, preprocessor, sample_data):
        """Test feature engineering"""
        df = sample_data.copy()
        df_engineered = preprocessor.create_engineered_features(df)
        
        # Check new features are created
        assert 'DEBT_TO_INCOME_RATIO' in df_engineered.columns
        assert 'CREDIT_TO_ANNUITY_RATIO' in df_engineered.columns
        
        # Check values are calculated correctly
        expected_dti = df['AMT_CREDIT'].iloc[0] / (df['AMT_INCOME_TOTAL'].iloc[0] + 1)
        actual_dti = df_engineered['DEBT_TO_INCOME_RATIO'].iloc[0]
        assert abs(expected_dti - actual_dti) < 0.001
    
    def test_scale_features(self, preprocessor, sample_data):
        """Test feature scaling"""
        df = sample_data.copy()
        df_scaled = preprocessor.scale_features(df, fit=True, exclude_cols=['TARGET', 'SK_ID_CURR'])
        
        # Check that numeric features are scaled (mean ~0, std ~1)
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['TARGET', 'SK_ID_CURR']]
        
        if len(numeric_cols) > 0:
            # For small datasets, just check scaling was applied
            assert df_scaled[numeric_cols].mean().abs().mean() < 2
    
    def test_fit_transform_pipeline(self, preprocessor, sample_data):
        """Test complete preprocessing pipeline"""
        df = sample_data.copy()
        
        X, y = preprocessor.fit_transform(df, target_col='TARGET')
        
        # Check shapes
        assert X.shape[0] == df.shape[0]
        assert X.shape[1] > df.shape[1] - 2  # More features due to engineering
        assert len(y) == df.shape[0]
        
        # Check no missing values
        assert not np.isnan(X).any()
        
        # Check target is correct
        assert np.array_equal(y, df['TARGET'].values)
    
    def test_transform_consistency(self, preprocessor, sample_data):
        """Test that transform produces consistent results"""
        df = sample_data.copy()
        
        # Fit on full data
        X1, y1 = preprocessor.fit_transform(df, target_col='TARGET')
        
        # Transform same data
        X2, y2 = preprocessor.transform(df, target_col='TARGET')
        
        # Check shapes match
        assert X1.shape == X2.shape
        assert len(y1) == len(y2)
    
    def test_save_and_load(self, preprocessor, sample_data, tmp_path):
        """Test saving and loading preprocessor"""
        df = sample_data.copy()
        
        # Fit preprocessor
        X, y = preprocessor.fit_transform(df, target_col='TARGET')
        
        # Save
        save_path = tmp_path / "test_preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Load
        loaded_preprocessor = CreditDataPreprocessor.load(str(save_path))
        
        # Transform with loaded preprocessor
        X_loaded, y_loaded = loaded_preprocessor.transform(df, target_col='TARGET')
        
        # Check results match
        assert X.shape == X_loaded.shape
        np.testing.assert_array_almost_equal(X, X_loaded, decimal=5)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self, preprocessor):
        """Test with empty dataframe"""
        df = pd.DataFrame()
        
        with pytest.raises(Exception):
            preprocessor.fit_transform(df)
    
    def test_missing_target(self, preprocessor, sample_data):
        """Test with missing target column"""
        df = sample_data.copy()
        df = df.drop(columns=['TARGET'])
        
        # Should work when target_col is specified correctly
        X = preprocessor.fit_transform(df, target_col='TARGET')
        assert X is not None
    
    def test_all_categorical(self, preprocessor):
        """Test with all categorical features"""
        df = pd.DataFrame({
            'TARGET': [0, 1, 0],
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z'],
        })
        
        X, y = preprocessor.fit_transform(df)
        assert X.shape[0] == 3
    
    def test_all_numeric(self, preprocessor):
        """Test with all numeric features"""
        df = pd.DataFrame({
            'TARGET': [0, 1, 0],
            'num1': [1.0, 2.0, 3.0],
            'num2': [10.0, 20.0, 30.0],
        })
        
        X, y = preprocessor.fit_transform(df)
        assert X.shape[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
