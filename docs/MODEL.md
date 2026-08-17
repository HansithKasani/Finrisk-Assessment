# Model Documentation

## Model Architecture

**Type**: XGBoost Gradient Boosting Classifier

### Hyperparameters

```python
{
    "n_estimators": 200,        # Number of boosting stages
    "max_depth": 6,             # Maximum tree depth
    "learning_rate": 0.1,       # Shrinkage parameter
    "subsample": 0.8,           # Training instance subsample ratio
    "colsample_bytree": 0.8,    # Feature subsample ratio
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42
}
```

## Training Data

- **Samples**: 307,511 loan applications
- **Features**: 85 total
  - Demographics: 9 features
  - Financial: 15+ features
  - Credit Bureau: 6 features
  - Engineered: 3 features
  - Other: 50+ features
- **Target**: Binary (Default=1, No Default=0)
- **Class Imbalance**: 8:1 ratio → Handled with SMOTE

## Data Preprocessing Pipeline

1. **Data Cleaning**
   - Missing value imputation (median for numerical, mode for categorical)
   - Outlier detection

2. **Feature Encoding**
   - OrdinalEncoder for categorical features
   - Handle unknown categories safely

3. **Feature Scaling**
   - StandardScaler for numerical features
   - Ensures consistent ranges

4. **Class Balancing**
   - SMOTE applied only to training set
   - Minority class oversampled to 1:1 ratio
   - Test set preserved without synthetic samples

## Model Performance

### Test Set Results

| Metric | Score |
|--------|-------|
| Accuracy | >85% |
| Precision | >80% |
| Recall | >75% |
| F1-Score | >75% |
| ROC-AUC | >0.85 |

### Decision Threshold

- **Default Threshold**: 0.50
- **Optimized Threshold**: 0.20
- **Rationale**: Minimize false negatives (high-risk applicants missed) at expense of higher false positives

## Feature Importance

Top predictive features:
1. Credit-to-Income Ratio
2. Annual Income
3. Previous Credit Inquiries
4. Number of Active Accounts
5. Credit Utilization Ratio

## Explainability

### SHAP Integration
- Global feature importance
- Individual prediction explanations
- Force plots and waterfall plots

### LIME Integration
- Local model approximations
- Feature contribution analysis

## Model Artifacts

### Files

- `credit_risk_xgboost_model.pkl` (794 KB)
  - Trained XGBoost model
  - Load with: `joblib.load("credit_risk_xgboost_model.pkl")`

- `credit_risk_encoder.pkl` (5.3 KB)
  - OrdinalEncoder for categorical features
  - Must be applied before model predictions

- `credit_risk_scaler.pkl` (Optional)
  - StandardScaler for feature normalization

### Using the Model

```python
import joblib

# Load artifacts
model = joblib.load("Models/credit_risk_xgboost_model.pkl")
encoder = joblib.load("Models/credit_risk_encoder.pkl")

# Transform new data
X_new = encoder.transform(X_new)

# Make predictions
probabilities = model.predict_proba(X_new)[:, 1]
predictions = (probabilities >= 0.20).astype(int)
```

## Monitoring & Updates

### Data Drift Detection
- Monitor feature distribution changes
- Track prediction distribution shifts
- Alert if model performance degrades below 80% accuracy

### Retraining Triggers
- Accuracy drops below 80%
- ROC-AUC drops below 0.80
- Significant feature distribution changes
- Regulatory requirement changes

## Ethical Considerations

- Fairness audit recommended
- No discrimination by protected attributes
- All decisions must be explainable
- GDPR/CCPA compliance required

