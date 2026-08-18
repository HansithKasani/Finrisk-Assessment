# 🚀 How to Use FinRisk-Assessment System

Complete guide to running and using your AI Credit Risk Assessment System.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (3 Steps)](#quick-start-3-steps)
3. [Using the Streamlit Dashboard](#using-the-streamlit-dashboard)
4. [Training Your Own Model](#training-your-own-model)
5. [Running Tests](#running-tests)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you start, ensure you have:

- ✅ **Python 3.8 or higher** installed
- ✅ **pip** package manager
- ✅ **8GB RAM** minimum (16GB recommended)
- ✅ **2GB free disk space**

Check your Python version:
```bash
python --version
# Should show: Python 3.8.x or higher
```

---

## Quick Start (3 Steps)

### Step 1: Install Dependencies

Open **Command Prompt** or **PowerShell** in your project folder:

```bash
# Navigate to project directory
cd "C:\Users\Hansith Kasani\Documents\AI-Credit-Risk-Assessment-System"

# Install all required packages
pip install -r requirements.txt
```

**This will take 2-5 minutes.** It installs:
- XGBoost, Scikit-learn (ML libraries)
- SHAP, LIME (Explainability)
- Streamlit (Dashboard)
- Pandas, NumPy (Data processing)
- Matplotlib, Plotly (Visualization)

### Step 2: Verify Installation

```bash
# Run dependency checker
python scripts/verify_dependencies.py
```

You should see ✅ marks for all packages.

### Step 3: Launch the Application

**Option A: Using the launcher (Easiest)**
```bash
python run_app.py
```

**Option B: Direct Streamlit command**
```bash
streamlit run App/main.py
```

**Option C: Windows batch file**
```bash
run.bat
```

Your browser will automatically open to: **http://localhost:8501**

---

## 🎨 Using the Streamlit Dashboard

### 1. **Access the Dashboard**

Once launched, you'll see the **AI Credit Risk Assessment Dashboard** in your browser.

### 2. **Enter Applicant Information**

In the **left sidebar**, fill in the following fields:

**Personal Information:**
- Gender: M / F / XNA
- Owns Car: Y / N
- Owns Real Estate: Y / N
- Number of Children: 0-20

**Financial Information:**
- Annual Income: e.g., $150,000
- Loan Amount Requested: e.g., $500,000
- Loan Annuity: e.g., $25,000
- Price of Goods: e.g., $450,000

**Employment Information:**
- Days Employed: e.g., -1000 (negative = employed for ~3 years)
- Age (in days): e.g., -14600 (negative = ~40 years old)

**Credit History:**
- External Source 1: 0.0 - 1.0 (credit bureau score)
- External Source 2: 0.0 - 1.0
- External Source 3: 0.0 - 1.0
- Region Rating: 1-3

### 3. **Submit Assessment**

Click the **"🔍 Assess Credit Risk"** button at the bottom of the sidebar.

### 4. **View Results**

The system will display:

**Risk Assessment:**
- ✅ **APPROVED** or ⚠️ **REJECTED**
- Default probability percentage
- Risk level (Very Low → Very High)

**Risk Gauge:**
- Visual gauge showing probability (0-100%)
- Color-coded risk zones

**Explanation:**
- SHAP-based feature importance
- Top contributing factors
- Detailed reasoning report

**Feature Impact Chart:**
- Bar chart showing which features affected the decision
- Color-coded by impact direction

### 5. **Detailed Report**

Expand **"📄 View Detailed Reasoning Report"** to see:
- Complete applicant profile summary
- Financial ratios calculated
- Credit bureau scores
- Risk assessment details
- Recommendation

---

## 🤖 Training Your Own Model

If you want to train a new model with your own data:

### Method 1: Using Jupyter Notebooks (Recommended for Learning)

```bash
# Start Jupyter
jupyter notebook

# Open and run notebooks in order:
# 1. Notebooks/01_EDA_Credit_Risk.ipynb       - Explore data
# 2. Notebooks/02_EDA_Credit_Risk.ipynb       - Feature engineering
# 3. Notebooks/03_Model_Training.ipynb        - Train model
```

### Method 2: Using Training Script (Quick)

```bash
# Basic training
python scripts/train_model.py --data Data/application_train.csv

# With SMOTE for class balancing
python scripts/train_model.py --data Data/application_train.csv --use-smote

# Specify output directory
python scripts/train_model.py --data Data/application_train.csv --output-dir Models --use-smote
```

**Training Options:**
```bash
--data          # Path to CSV file
--model         # xgboost or random_forest
--output-dir    # Where to save models
--use-smote     # Balance classes
--test-size     # Test set proportion (default: 0.2)
--random-state  # Random seed (default: 42)
```

**Training Time:**
- Without GPU: 10-20 minutes
- With GPU: 3-5 minutes

---

## 🧪 Running Tests

Verify everything works correctly:

### Run All Tests

```bash
# Basic test run
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=html

# Using test runner script
python scripts/run_tests.py
```

### Run Specific Tests

```bash
# Test preprocessing
pytest tests/test_preprocessor.py -v

# Test explainability
pytest tests/test_explainer.py -v

# Test metrics
pytest tests/test_metrics.py -v

# Test model
pytest tests/test_model.py -v
```

### View Coverage Report

After running tests with coverage:
```bash
# Windows
start htmlcov\index.html

# Linux/Mac
open htmlcov/index.html
```

---

## 🎓 Advanced Usage

### 1. **Programmatic API Usage**

Use the system in your Python code:

```python
import pandas as pd
import joblib
from src.data.preprocessor import CreditDataPreprocessor
from src.models.explainer import CreditRiskExplainer

# Load trained model
model = joblib.load('Notebooks/credit_risk_xgboost_model.pkl')

# Load preprocessor (or create new)
preprocessor = CreditDataPreprocessor()

# Prepare sample data
applicant_data = {
    'CODE_GENDER': 'M',
    'FLAG_OWN_CAR': 'Y',
    'AMT_INCOME_TOTAL': 150000,
    'AMT_CREDIT': 500000,
    # ... other features
}
df = pd.DataFrame([applicant_data])

# Preprocess
X, _ = preprocessor.fit_transform(df, target_col='TARGET')

# Predict
probability = model.predict_proba(X)[0][1]
decision = "REJECT" if probability >= 0.5 else "APPROVE"

print(f"Decision: {decision}")
print(f"Default Probability: {probability:.2%}")

# Get explanation
explainer = CreditRiskExplainer(model, X.columns.tolist(), X)
report = explainer.generate_reasoning_report(X.values[0], probability)
print(report)
```

### 2. **Batch Processing**

Process multiple applicants at once:

```python
import pandas as pd
import joblib

# Load model
model = joblib.load('Notebooks/credit_risk_xgboost_model.pkl')

# Load batch of applicants
applicants = pd.read_csv('new_applicants.csv')

# Preprocess
from src.data.preprocessor import CreditDataPreprocessor
preprocessor = CreditDataPreprocessor()
X, _ = preprocessor.fit_transform(applicants, target_col='TARGET')

# Predict for all
probabilities = model.predict_proba(X)[:, 1]
decisions = ['REJECT' if p >= 0.5 else 'APPROVE' for p in probabilities]

# Create results dataframe
results = pd.DataFrame({
    'applicant_id': applicants['SK_ID_CURR'],
    'decision': decisions,
    'probability': probabilities
})

# Save results
results.to_csv('assessment_results.csv', index=False)
print(f"Processed {len(results)} applicants")
```

### 3. **Custom Model Training**

Train with custom parameters:

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and prepare data
df = pd.read_csv('Data/application_train.csv')
from src.data.preprocessor import CreditDataPreprocessor
preprocessor = CreditDataPreprocessor()
X, y = preprocessor.fit_transform(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train custom model
model = XGBClassifier(
    n_estimators=300,      # More trees
    max_depth=8,           # Deeper trees
    learning_rate=0.05,    # Slower learning
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
from src.utils.metrics import evaluate_model
evaluator = evaluate_model(
    y_test, 
    model.predict_proba(X_test)[:, 1],
    save_dir='reports/custom_model'
)

# Save model
import joblib
joblib.dump(model, 'Models/custom_model.pkl')
```

### 4. **Export Predictions to Excel**

```python
import pandas as pd

# After getting predictions
results = pd.DataFrame({
    'Applicant_ID': df['SK_ID_CURR'],
    'Decision': decisions,
    'Probability': probabilities,
    'Risk_Level': risk_levels,
    'Top_Factor_1': top_factors_1,
    'Top_Factor_2': top_factors_2,
    'Top_Factor_3': top_factors_3
})

# Export to Excel
results.to_excel('credit_risk_assessment_results.xlsx', index=False)
```

---

## 🐳 Using Docker

### Build and Run with Docker

```bash
# Build Docker image
docker build -t finrisk-assessment .

# Run container
docker run -p 8501:8501 finrisk-assessment

# Access at: http://localhost:8501
```

### Using Docker Compose

```bash
# Start services
docker-compose up

# Run in background
docker-compose up -d

# Stop services
docker-compose down
```

---

## 🛠️ Troubleshooting

### Issue: "Module not found"

**Solution:**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Issue: "Streamlit not launching"

**Solution:**
```bash
# Check if port 8501 is in use
netstat -ano | findstr :8501

# Use different port
streamlit run App/main.py --server.port 8080
```

### Issue: "Model file not found"

**Solution:**
The trained models are in `Notebooks/` directory:
```python
# Update path in your code
model = joblib.load('Notebooks/credit_risk_xgboost_model.pkl')
```

Or train your own:
```bash
python scripts/train_model.py --data Data/application_train.csv
```

### Issue: "Out of memory during training"

**Solution:**
```python
# Use a sample of data
df = pd.read_csv('Data/application_train.csv', nrows=50000)
```

### Issue: "SHAP taking too long"

**Solution:**
```python
# Use a smaller background sample
explainer = CreditRiskExplainer(
    model, 
    feature_names, 
    X_train=X_train[:100]  # Use only 100 samples
)
```

---

## 📊 Understanding the Output

### Risk Levels

| Probability | Risk Level | Decision | Description |
|-------------|-----------|----------|-------------|
| 0-15% | 🟢 Very Low | APPROVE | Excellent candidate |
| 15-30% | 🟡 Low | APPROVE | Good candidate |
| 30-50% | 🟠 Medium | APPROVE/REJECT | Borderline case |
| 50-70% | 🔴 High | REJECT | Risky candidate |
| 70-100% | ⛔ Very High | REJECT | Very risky |

### Feature Impact

**Positive Impact (Increases Risk):**
- High debt-to-income ratio
- Low credit bureau scores
- High loan amount relative to income
- Short employment history
- Recent credit inquiries

**Negative Impact (Decreases Risk):**
- High income
- Good credit scores
- Long employment history
- Low credit utilization
- Stable residential status

---

## 🎯 Example Use Cases

### Use Case 1: Single Applicant Assessment

```bash
1. Launch dashboard: streamlit run App/main.py
2. Enter applicant details in sidebar
3. Click "Assess Credit Risk"
4. Review decision and explanation
5. Download/print detailed report
```

### Use Case 2: Batch Assessment

```bash
1. Prepare CSV with applicant data
2. Run: python scripts/batch_assess.py --input applicants.csv
3. Get results in: results/assessments.csv
4. Review explanations for rejected applications
```

### Use Case 3: Model Comparison

```bash
1. Train multiple models with different parameters
2. Evaluate each on test set
3. Compare ROC-AUC, Precision, Recall
4. Select best model for deployment
5. Document decision rationale
```

### Use Case 4: Bias Testing

```bash
1. Load test data
2. Split by demographic groups
3. Calculate approval rates per group
4. Test for disparate impact
5. Document findings and mitigation
```

---

## 📚 Additional Resources

**Documentation:**
- [INSTALL.md](INSTALL.md) - Detailed installation guide
- [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) - Technical details
- [docs/API.md](docs/API.md) - API documentation
- [docs/MODEL.md](docs/MODEL.md) - Model specifications

**Notebooks:**
- `Notebooks/01_EDA_Credit_Risk.ipynb` - Explore the data
- `Notebooks/02_EDA_Credit_Risk.ipynb` - Feature engineering
- `Notebooks/03_Model_Training.ipynb` - Model training

**GitHub:**
- Repository: https://github.com/HansithKasani/Finrisk-Assessment
- Issues: Report bugs or request features
- Wiki: Community documentation

---

## 🎓 Learning Path

**Beginner:**
1. ✅ Install and run the dashboard
2. ✅ Make a few test predictions
3. ✅ Read the SHAP explanations
4. ✅ Understand the features

**Intermediate:**
1. ✅ Run the Jupyter notebooks
2. ✅ Train your own model
3. ✅ Experiment with parameters
4. ✅ Run the test suite

**Advanced:**
1. ✅ Integrate into existing systems
2. ✅ Deploy to production
3. ✅ Conduct bias testing
4. ✅ Implement monitoring
5. ✅ Add custom features

---

## ⚖️ Ethical Usage Reminder

**Always Remember:**
- ✅ This is a decision **support** tool, not an autonomous decision maker
- ✅ Human review is required for all decisions
- ✅ Comply with fair lending regulations
- ✅ Regular bias audits are necessary
- ✅ Provide clear explanations to applicants
- ✅ Allow for appeals and overrides

---

## 💡 Pro Tips

1. **Start Simple:** Begin with the dashboard before diving into code
2. **Use Test Data:** Don't use real customer data during testing
3. **Check Logs:** Review `logs/` directory for detailed information
4. **Save Results:** Export important assessments for records
5. **Regular Updates:** Retrain model quarterly with fresh data
6. **Monitor Performance:** Track prediction accuracy over time
7. **Document Decisions:** Keep audit trail of all overrides

---

## 🆘 Getting Help

**If you encounter issues:**

1. Check [Troubleshooting](#troubleshooting) section
2. Review logs in `logs/` directory
3. Run tests: `pytest tests/ -v`
4. Check GitHub Issues: https://github.com/HansithKasani/Finrisk-Assessment/issues
5. Consult documentation in `docs/` folder

---

## 🎉 You're Ready!

You now know how to:
- ✅ Install and launch the system
- ✅ Use the interactive dashboard
- ✅ Train custom models
- ✅ Run tests and verify functionality
- ✅ Interpret results and explanations
- ✅ Integrate into your workflow

**Start with:** `python run_app.py`

**Have fun assessing credit risk! 🚀**

---

**Last Updated:** August 2026  
**Version:** 1.0.0  
**Support:** https://github.com/HansithKasani/Finrisk-Assessment
