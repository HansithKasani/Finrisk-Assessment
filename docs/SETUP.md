# Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip or conda
- Git
- 4GB+ RAM for model training
- 2GB+ disk space

## Installation Steps

### 1. Clone/Navigate to Repository

```bash
cd AI-Credit-Risk-Assessment-System
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import pandas, xgboost, streamlit; print('✅ All packages installed!')"
```

### 5. Run Jupyter Notebooks

```bash
jupyter notebook Notebooks/
```

Open and run:
- `01_EDA_Credit_Risk.ipynb` - Exploratory Data Analysis
- `02_Feature_Engineering_Credit_Risk.ipynb` - Feature Engineering
- `03_Model_Training_Credit_Risk.ipynb` - Model Training

### 6. Run Tests

```bash
pytest tests/ -v
```

### 7. Docker Setup (Optional)

```bash
# Build Docker image
docker build -t credit-risk-ai .

# Run with Docker Compose
docker-compose up
```

The app will be available at `http://localhost:8501`

## Troubleshooting

### Issue: Module import errors

```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: Jupyter not found

```bash
pip install jupyter notebook
```

### Issue: XGBoost/SHAP errors

```bash
pip install --upgrade xgboost shap
```

### Issue: Data not found

Ensure CSV files are in `Data/raw/` directory

## Next Steps

1. Review notebooks for complete workflow
2. Check `reports/` folder for analysis
3. Review model artifacts in `Models/` folder
4. Run Streamlit app when ready: `streamlit run App/main.py`

