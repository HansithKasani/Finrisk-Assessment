"""
Model training script for Credit Risk Assessment
Can be run from command line to train and save model
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import CreditDataPreprocessor
from src.utils.metrics import evaluate_model
from src.utils.logger import get_logger

logger = get_logger("ModelTraining")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Credit Risk Assessment Model"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="Data/application_train.csv",
        help="Path to training data CSV"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "random_forest"],
        default="xgboost",
        help="Model type to train"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Models",
        help="Directory to save trained models"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (0.0-1.0)"
    )
    
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Apply SMOTE for class imbalance"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load training data"""
    logger.info(f"Loading data from {data_path}")
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def preprocess_data(df: pd.DataFrame, preprocessor: CreditDataPreprocessor = None):
    """Preprocess data for training"""
    logger.info("Preprocessing data...")
    
    if preprocessor is None:
        preprocessor = CreditDataPreprocessor()
    
    X, y = preprocessor.fit_transform(df, target_col='TARGET')
    
    logger.info(f"Preprocessed data shape: {X.shape}")
    logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y, preprocessor


def train_model(X_train, y_train, model_type: str, use_gpu: bool = False):
    """Train the model"""
    logger.info(f"Training {model_type} model...")
    
    if model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='gpu_hist' if use_gpu else 'hist',
            random_state=42,
            eval_metric='auc'
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    
    return model


def save_artifacts(model, preprocessor, output_dir: str, model_type: str):
    """Save trained model and preprocessor"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"credit_risk_{model_type}_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save preprocessor
    preprocessor_path = output_dir / "credit_risk_preprocessor.pkl"
    preprocessor.save(str(preprocessor_path))
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    return model_path, preprocessor_path


def main():
    """Main training pipeline"""
    args = parse_args()
    
    logger.info("="*70)
    logger.info("Credit Risk Model Training Pipeline")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Test Size: {args.test_size}")
    logger.info(f"  Use SMOTE: {args.use_smote}")
    logger.info(f"  Random State: {args.random_state}")
    logger.info("="*70)
    
    try:
        # Load data
        df = load_data(args.data)
        
        # Preprocess
        X, y, preprocessor = preprocess_data(df)
        
        # Split data
        logger.info(f"Splitting data (test size: {args.test_size})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Apply SMOTE if requested
        if args.use_smote:
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=args.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {X_train.shape[0]} samples")
            logger.info(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        # Train model
        model = train_model(X_train, y_train, args.model)
        
        # Evaluate
        logger.info("Evaluating model on test set...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        evaluation_dir = Path(args.output_dir).parent / "reports" / "evaluation"
        evaluator = evaluate_model(
            y_test, 
            y_pred_proba, 
            threshold=0.5,
            save_dir=str(evaluation_dir)
        )
        
        # Print summary
        metrics = evaluator.calculate_all_metrics()
        logger.info("\nModel Performance:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Save model and preprocessor
        model_path, prep_path = save_artifacts(
            model, 
            preprocessor, 
            args.output_dir,
            args.model
        )
        
        logger.info("\n" + "="*70)
        logger.info("✅ Training Complete!")
        logger.info("="*70)
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Preprocessor saved: {prep_path}")
        logger.info(f"Evaluation reports: {evaluation_dir}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
