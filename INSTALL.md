# Installation & Setup Guide

Complete guide for setting up the AI Credit Risk Assessment System.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup](#quick-setup)
3. [Manual Setup](#manual-setup)
4. [Data Setup](#data-setup)
5. [Model Training](#model-training)
6. [Running the Application](#running-the-application)
7. [Docker Deployment](#docker-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Git** (optional, for cloning repository)

### System Requirements

- **OS**: Windows 10/11, macOS 10.14+, or Linux
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 2GB for data and models
- **GPU** (optional): NVIDIA GPU with CUDA support for faster training

### Check Prerequisites

```bash
# Check Python version
python --version
# Should show: Python 3.8.x or higher

# Check pip
pip --version

# Check Git (optional)
git --version
```

---

## Quick Setup

### Windows

1. **Run the automated setup:**
   ```cmd
   setup.bat
   ```

2. **Follow the prompts** to install dependencies

3. **Download data** (see [Data Setup](#data-setup))

4. **Run the application:**
   ```cmd
   run.bat
   ```

### Linux/Mac

1. **Run the setup script:**
   ```bash
   python scripts/setup_environment.py
   ```

2. **Download data** (see [Data Setup](#data-setup))

3. **Run the application:**
   ```bash
   python run_app.py
   # or
   streamlit run App/main.py
   ```

---

## Manual Setup

### Step 1: Clone or Download

**Option A: Clone with Git**
```bash
git clone https://github.com/yourusername/AI-Credit-Risk-Assessment-System.git
cd AI-Credit-Risk-Assessment-System
```

**Option B: Download ZIP**
- Download from GitHub
- Extract to desired location
- Open terminal in extracted folder

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- Core: numpy, pandas, scikit-learn
- ML: xgboost, lightgbm, imbalanced-learn
- Explainability: shap, lime
- UI: streamlit
- Visualization: matplotlib, seaborn, plotly
- Testing: pytest
- And more...

### Step 4: Create Directory Structure

The setup script creates these automatically, but you can create manually:

```bash
mkdir -p Data/raw Data/processed
mkdir -p Models
mkdir -p logs
mkdir -p reports/figures reports/results reports/evaluation
```

### Step 5: Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy from example
cp config/.env.example .env

# Edit with your settings
nano .env  # or use any text editor
```

---

## Data Setup

### Download Training Data

The system requires the **Home Credit Default Risk** dataset from Kaggle.

**Option 1: Kaggle API (Recommended)**

```bash
# Install Kaggle API
pip install kaggle

# Setup Kaggle credentials (~/.kaggle/kaggle.json)
# Download from: https://www.kaggle.com/account

# Download dataset
kaggle competitions download -c home-credit-default-risk

# Extract
unzip home-credit-default-risk.zip -d Data/
```

**Option 2: Manual Download**

1. Visit: https://www.kaggle.com/c/home-credit-default-risk/data
2. Download `application_train.csv`
3. Place in `Data/application_train.csv`

### Verify Data

```bash
# Check if data exists
python -c "import pandas as pd; df = pd.read_csv('Data/application_train.csv'); print(f'✓ Loaded {len(df)} rows, {len(df.columns)} columns')"
```

Expected output:
```
✓ Loaded 307511 rows, 122 columns
```

---

## Model Training

### Option 1: Use Pre-trained Models

If you have pre-trained models:

1. Place files in `Models/` or `Notebooks/` directory:
   - `credit_risk_xgboost_model.pkl`
   - `credit_risk_encoder.pkl`
   - `credit_risk_preprocessor.pkl`

2. Verify:
   ```bash
   python -c "import joblib; model = joblib.load('Models/credit_risk_xgboost_model.pkl'); print('✓ Model loaded')"
   ```

### Option 2: Train from Scratch

**Method A: Using Jupyter Notebooks (Recommended for Learning)**

```bash
# Start Jupyter
jupyter notebook

# Run notebooks in order:
# 1. Notebooks/01_EDA_Credit_Risk.ipynb
# 2. Notebooks/02_EDA_Credit_Risk.ipynb
# 3. Notebooks/03_Model_Training.ipynb
```

**Method B: Using Training Script (Quick)**

```bash
# Basic training
python scripts/train_model.py --data Data/application_train.csv

# With SMOTE and custom output
python scripts/train_model.py \
    --data Data/application_train.csv \
    --model xgboost \
    --use-smote \
    --output-dir Models
```

**Training Options:**
- `--data`: Path to training data
- `--model`: Model type (xgboost or random_forest)
- `--output-dir`: Where to save models
- `--use-smote`: Apply SMOTE for class imbalance
- `--test-size`: Test set size (default: 0.2)

**Training Time:**
- Without GPU: 10-20 minutes
- With GPU: 3-5 minutes

### Verify Trained Model

```bash
# Check model files
ls -lh Models/

# Should see:
# - credit_risk_xgboost_model.pkl
# - credit_risk_preprocessor.pkl
```

---

## Running the Application

### Method 1: Using Launcher Script

```bash
# Windows
run.bat

# Linux/Mac
python run_app.py
```

### Method 2: Direct Streamlit

```bash
streamlit run App/main.py
```

### Method 3: Custom Port

```bash
streamlit run App/main.py --server.port 8080
```

### Access the Application

1. Open browser to: **http://localhost:8501**
2. Fill in applicant information in the sidebar
3. Click "Assess Credit Risk"
4. View prediction and explanation

---

## Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t credit-risk-ai:latest .

# Verify build
docker images | grep credit-risk-ai
```

### Run Container

```bash
# Run with default settings
docker run -p 8501:8501 credit-risk-ai:latest

# Run with volume mount (for models)
docker run -p 8501:8501 -v $(pwd)/Models:/app/Models credit-risk-ai:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Run Specific Test

```bash
pytest tests/test_model.py -v
```

---

## Troubleshooting

### Issue: Python not found

**Solution:**
- Install Python 3.8+ from [python.org](https://www.python.org/)
- Ensure "Add Python to PATH" is checked during installation
- Restart terminal after installation

### Issue: pip not found

**Solution:**
```bash
# Windows
python -m ensurepip --upgrade

# Linux
sudo apt-get install python3-pip
```

### Issue: Module not found

**Solution:**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Or install specific package
pip install package-name
```

### Issue: CUDA not available (for GPU training)

**Solution:**
- Install NVIDIA CUDA Toolkit
- Install compatible GPU drivers
- Reinstall xgboost with GPU support:
  ```bash
  pip install xgboost --upgrade
  ```

### Issue: Streamlit not launching

**Solution:**
```bash
# Check if port is in use
netstat -ano | findstr :8501

# Use different port
streamlit run App/main.py --server.port 8080

# Clear streamlit cache
streamlit cache clear
```

### Issue: Model file not found

**Solution:**
- Verify model files exist in `Models/` or `Notebooks/` directory
- Check file paths in `.env` file
- Retrain model using notebooks or training script

### Issue: Out of memory during training

**Solution:**
- Reduce dataset size for testing:
  ```python
  df = pd.read_csv('Data/application_train.csv', nrows=50000)
  ```
- Close other applications
- Use smaller batch sizes
- Train on machine with more RAM

### Issue: Slow performance

**Solution:**
- Use GPU for training (if available)
- Reduce model complexity (fewer estimators)
- Use sampling during development
- Close unnecessary applications

---

## Additional Configuration

### Environment Variables

Edit `.env` file to customize:

```ini
MODEL_PATH=Models/credit_risk_xgboost_model.pkl
PREPROCESSOR_PATH=Models/credit_risk_preprocessor.pkl
APP_PORT=8501
LOG_LEVEL=INFO
PREDICTION_THRESHOLD=0.5
```

### Logging

Logs are saved in `logs/` directory:
- `credit_risk_YYYYMMDD.log` - All logs
- `credit_risk_errors_YYYYMMDD.log` - Errors only

Configure log level in `.env`:
```ini
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

---

## Getting Help

**Documentation:**
- 📖 [README.md](README.md) - Project overview
- 📊 [MODEL.md](docs/MODEL.md) - Model details
- 🔧 [SETUP.md](docs/SETUP.md) - Additional setup info
- 🚀 [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Production deployment

**Issues:**
- Check existing issues on GitHub
- Create new issue with:
  - Python version
  - Operating system
  - Error messages
  - Steps to reproduce

**Contact:**
- Email: your.email@example.com
- GitHub: @yourusername

---

## Next Steps

After successful installation:

1. ✅ **Explore the data** - Run EDA notebooks
2. ✅ **Train a model** - Use training notebooks or script
3. ✅ **Test the app** - Launch Streamlit dashboard
4. ✅ **Read documentation** - Understand the system
5. ✅ **Review ethics guide** - Responsible AI usage

---

**Last Updated:** August 2026  
**Version:** 1.0.0
