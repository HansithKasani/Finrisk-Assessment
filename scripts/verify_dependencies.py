"""
Verify all required dependencies are installed and compatible
"""

import sys
import importlib
import subprocess
from pathlib import Path
from typing import List, Tuple


# Required packages with minimum versions
REQUIRED_PACKAGES = {
    'numpy': '1.24.0',
    'pandas': '2.0.0',
    'scipy': '1.11.0',
    'sklearn': '1.3.0',
    'xgboost': '2.0.0',
    'lightgbm': '4.0.0',
    'imblearn': '0.11.0',
    'shap': '0.42.0',
    'lime': '0.2.0',
    'streamlit': '1.26.0',
    'matplotlib': '3.7.0',
    'seaborn': '0.12.0',
    'plotly': '5.16.0',
    'joblib': '1.3.0',
    'yaml': '6.0.0',
    'dotenv': '1.0.0',
    'pytest': '7.4.0',
}


def get_version_tuple(version_string: str) -> Tuple[int, ...]:
    """Convert version string to tuple for comparison"""
    try:
        return tuple(map(int, version_string.split('.')[:3]))
    except:
        return (0, 0, 0)


def check_package(package_name: str, min_version: str) -> Tuple[bool, str, str]:
    """
    Check if a package is installed and meets minimum version
    
    Returns:
        (is_installed, installed_version, status_message)
    """
    try:
        # Special case for sklearn
        if package_name == 'sklearn':
            module = importlib.import_module('sklearn')
        # Special case for yaml
        elif package_name == 'yaml':
            module = importlib.import_module('yaml')
        # Special case for dotenv
        elif package_name == 'dotenv':
            module = importlib.import_module('dotenv')
        else:
            module = importlib.import_module(package_name)
        
        # Get version
        if hasattr(module, '__version__'):
            installed_version = module.__version__
        elif hasattr(module, 'VERSION'):
            installed_version = module.VERSION
        else:
            installed_version = "unknown"
        
        # Compare versions
        if installed_version != "unknown":
            installed_tuple = get_version_tuple(installed_version)
            required_tuple = get_version_tuple(min_version)
            
            if installed_tuple >= required_tuple:
                return True, installed_version, "OK"
            else:
                return True, installed_version, f"VERSION TOO OLD (need {min_version}+)"
        else:
            return True, installed_version, "VERSION UNKNOWN"
            
    except ImportError:
        return False, "not installed", "NOT INSTALLED"


def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check")
    print("=" * 70)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Current Python: {version_str}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ FAIL: Python 3.8+ required")
        return False
    else:
        print("✅ PASS: Python version compatible")
        return True


def check_all_packages():
    """Check all required packages"""
    print("\n📦 Package Dependency Check")
    print("=" * 70)
    
    results = []
    all_ok = True
    
    for package_name, min_version in REQUIRED_PACKAGES.items():
        is_installed, installed_version, status = check_package(package_name, min_version)
        results.append((package_name, is_installed, installed_version, status))
        
        # Print status
        status_symbol = "✅" if status == "OK" else "⚠️" if "VERSION" in status else "❌"
        print(f"{status_symbol} {package_name:20s} {installed_version:15s} {status}")
        
        if status != "OK":
            all_ok = False
    
    return all_ok, results


def check_gpu_support():
    """Check if GPU support is available"""
    print("\n🎮 GPU Support Check")
    print("=" * 70)
    
    try:
        import xgboost as xgb
        
        # Try to use GPU
        try:
            # Create a small test dataset
            import numpy as np
            from xgboost import DMatrix
            
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)
            dtrain = DMatrix(X, label=y)
            
            # Try GPU training
            params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
            xgb.train(params, dtrain, num_boost_round=1)
            
            print("✅ GPU support is AVAILABLE and WORKING")
            return True
        except Exception as e:
            print(f"⚠️  GPU support not available: {str(e)[:50]}")
            print("   Training will use CPU (slower but functional)")
            return False
    except ImportError:
        print("❌ XGBoost not installed")
        return False


def generate_installation_commands(results: List[Tuple]):
    """Generate pip install commands for missing packages"""
    print("\n🔧 Installation Commands for Missing/Outdated Packages")
    print("=" * 70)
    
    missing_packages = []
    outdated_packages = []
    
    for package_name, is_installed, installed_version, status in results:
        if status == "NOT INSTALLED":
            missing_packages.append(package_name)
        elif "VERSION TOO OLD" in status:
            outdated_packages.append(package_name)
    
    if missing_packages or outdated_packages:
        print("\nRun the following commands:\n")
        
        if missing_packages or outdated_packages:
            print("# Install/Update all dependencies:")
            print("pip install --upgrade -r requirements.txt")
            print()
        
        if missing_packages:
            print("# Or install missing packages individually:")
            for package in missing_packages:
                # Map internal names to pip names
                pip_name = package
                if package == 'sklearn':
                    pip_name = 'scikit-learn'
                elif package == 'yaml':
                    pip_name = 'pyyaml'
                elif package == 'dotenv':
                    pip_name = 'python-dotenv'
                
                print(f"pip install {pip_name}")
    else:
        print("✅ All dependencies are satisfied!")


def main():
    """Main verification function"""
    print("\n" + "=" * 70)
    print("AI CREDIT RISK ASSESSMENT - DEPENDENCY VERIFICATION")
    print("=" * 70 + "\n")
    
    # Check Python version
    python_ok = check_python_version()
    
    if not python_ok:
        print("\n❌ CRITICAL: Python version incompatible")
        print("Please install Python 3.8 or higher")
        sys.exit(1)
    
    # Check packages
    packages_ok, results = check_all_packages()
    
    # Check GPU support (optional)
    check_gpu_support()
    
    # Generate installation commands if needed
    if not packages_ok:
        generate_installation_commands(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if packages_ok:
        print("✅ All dependencies are installed and compatible!")
        print("\nYou can now:")
        print("  - Run the application: python run_app.py")
        print("  - Train models: python scripts/train_model.py")
        print("  - Run tests: pytest tests/ -v")
    else:
        print("⚠️  Some dependencies are missing or outdated")
        print("\nPlease install the required packages:")
        print("  pip install -r requirements.txt")
        print("\nThen run this script again to verify.")
    
    print("=" * 70 + "\n")
    
    return 0 if packages_ok else 1


if __name__ == "__main__":
    sys.exit(main())
