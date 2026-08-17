"""
Environment setup script for Credit Risk Assessment System
Creates necessary directories and checks dependencies
"""

import os
import sys
from pathlib import Path
import subprocess

def create_directories():
    """Create necessary project directories"""
    print("📁 Creating project directories...")
    
    base_dir = Path(__file__).parent.parent
    
    directories = [
        base_dir / "Data" / "raw",
        base_dir / "Data" / "processed",
        base_dir / "Models",
        base_dir / "logs",
        base_dir / "reports" / "figures",
        base_dir / "reports" / "results",
        base_dir / "reports" / "evaluation",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("✅ Directories created successfully\n")

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, but found {version.major}.{version.minor}")
        return False
    
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    print("✅ Python version compatible\n")
    return True

def check_pip():
    """Check if pip is available"""
    print("📦 Checking pip installation...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  ✓ {result.stdout.strip()}")
        print("✅ pip is available\n")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip not found")
        return False

def install_requirements():
    """Install required packages"""
    print("📥 Installing dependencies...")
    
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ requirements.txt not found at {requirements_file}")
        return False
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True
        )
        print("✅ Dependencies installed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_env_file():
    """Create .env file template if it doesn't exist"""
    print("⚙️  Checking environment configuration...")
    
    base_dir = Path(__file__).parent.parent
    env_file = base_dir / ".env"
    env_example = base_dir / "config" / ".env.example"
    
    if not env_file.exists():
        if env_example.exists():
            print(f"  📝 Creating .env from {env_example.name}")
            env_file.write_text(env_example.read_text())
            print(f"  ✓ Created .env file at {env_file}")
        else:
            # Create basic .env template
            env_content = """# Environment Configuration for Credit Risk Assessment System

# Model Configuration
MODEL_PATH=Models/credit_risk_xgboost_model.pkl
PREPROCESSOR_PATH=Models/credit_risk_preprocessor.pkl
ENCODER_PATH=Notebooks/credit_risk_encoder.pkl

# Application Configuration
APP_HOST=localhost
APP_PORT=8501
LOG_LEVEL=INFO

# Data Configuration
DATA_PATH=Data/application_train.csv
PROCESSED_DATA_PATH=Data/application_train_cleaned.csv

# Model Parameters
PREDICTION_THRESHOLD=0.5
BATCH_SIZE=1000

# Logging
LOG_DIR=logs/
"""
            env_file.write_text(env_content)
            print(f"  ✓ Created default .env file at {env_file}")
        
        print("  ⚠️  Please review and update .env with your configuration")
    else:
        print(f"  ✓ .env file exists at {env_file}")
    
    print("✅ Environment configuration ready\n")

def verify_data():
    """Check if data files exist"""
    print("📊 Checking for data files...")
    
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "Data" / "application_train.csv"
    
    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Found {data_file.name} ({size_mb:.1f} MB)")
        print("✅ Training data available\n")
    else:
        print(f"  ⚠️  Training data not found at {data_file}")
        print("  ℹ️  Download from: https://www.kaggle.com/c/home-credit-default-risk/data")
        print("  ℹ️  Place in: Data/application_train.csv\n")

def verify_models():
    """Check if trained models exist"""
    print("🤖 Checking for trained models...")
    
    base_dir = Path(__file__).parent.parent
    
    model_files = [
        base_dir / "Notebooks" / "credit_risk_xgboost_model.pkl",
        base_dir / "Models" / "credit_risk_xgboost_model.pkl",
        base_dir / "Notebooks" / "credit_risk_encoder.pkl",
    ]
    
    found_models = []
    for model_file in model_files:
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ Found {model_file.name} ({size_mb:.1f} MB)")
            found_models.append(model_file)
    
    if found_models:
        print("✅ Trained models available\n")
    else:
        print("  ⚠️  No trained models found")
        print("  ℹ️  Train a model by running: 03_Model_Training.ipynb")
        print("  ℹ️  Or place pre-trained models in Models/ directory\n")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*70)
    print("🎉 Setup Complete!")
    print("="*70)
    print("\n📋 Next Steps:\n")
    print("1. 📊 Ensure data is available:")
    print("   - Place application_train.csv in Data/ directory")
    print("   - Download from Kaggle if needed\n")
    
    print("2. 🤖 Train the model (if not done already):")
    print("   - Open Jupyter: jupyter notebook")
    print("   - Run notebooks in order: 01, 02, 03\n")
    
    print("3. 🚀 Launch the application:")
    print("   - Run: python run_app.py")
    print("   - Or: streamlit run App/main.py\n")
    
    print("4. 🧪 Run tests:")
    print("   - Run: pytest tests/ -v\n")
    
    print("5. 📖 Read the documentation:")
    print("   - Check docs/ folder for detailed guides")
    print("   - Review README.md for project overview\n")
    
    print("="*70)
    print("💡 For help: Check README.md or open an issue on GitHub")
    print("="*70 + "\n")

def main():
    """Main setup function"""
    print("\n" + "="*70)
    print("🔧 AI Credit Risk Assessment System - Environment Setup")
    print("="*70 + "\n")
    
    # Run setup steps
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    create_directories()
    
    # Ask user if they want to install dependencies
    response = input("📥 Install Python dependencies? (y/n): ").strip().lower()
    if response == 'y':
        if not install_requirements():
            print("\n⚠️  Dependency installation failed. You may need to install manually.")
    else:
        print("⏭️  Skipping dependency installation\n")
    
    create_env_file()
    verify_data()
    verify_models()
    
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        sys.exit(1)
