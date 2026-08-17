# Test Suite Documentation

Comprehensive test suite for the AI Credit Risk Assessment System.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)

---

## Overview

This test suite ensures the reliability and correctness of the Credit Risk Assessment System through:

- **Unit Tests**: Testing individual components in isolation
- **Integration Tests**: Testing component interactions
- **Edge Case Tests**: Testing boundary conditions and error handling
- **Performance Tests**: Ensuring acceptable performance

---

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── test_model.py              # Model training and prediction tests
├── test_preprocessor.py       # Data preprocessing tests
├── test_explainer.py          # Explainability module tests
├── test_metrics.py            # Evaluation metrics tests
└── README.md                  # This file
```

### Test Files

#### `test_preprocessor.py`
Tests for data preprocessing pipeline:
- Feature type identification
- Missing value handling
- Categorical encoding
- Feature engineering
- Feature scaling
- Pipeline consistency
- Save/load functionality

#### `test_explainer.py`
Tests for model explainability:
- SHAP/LIME explainer initialization
- Reasoning report generation
- Feature importance extraction
- Risk level classification
- Edge cases and error handling

#### `test_metrics.py`
Tests for evaluation metrics:
- Metric calculation (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix generation
- Optimal threshold finding
- Report generation
- Various data scenarios

#### `test_model.py`
Tests for model training and inference:
- Data loading and preprocessing
- Model training
- Prediction generation
- Feature engineering
- Performance validation
- Model serialization

---

## Running Tests

### Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessor.py -v

# Run specific test class
pytest tests/test_preprocessor.py::TestCreditDataPreprocessor -v

# Run specific test function
pytest tests/test_preprocessor.py::TestCreditDataPreprocessor::test_initialization -v
```

### Using Test Runner Script

```bash
# Run all tests with coverage
python scripts/run_tests.py

# Run only unit tests
python scripts/run_tests.py --mode unit

# Run quick tests (exclude slow)
python scripts/run_tests.py --mode quick

# Run specific file
python scripts/run_tests.py --file test_preprocessor.py

# Run without coverage (faster)
python scripts/run_tests.py --no-coverage
```

### Using Batch Script (Windows)

```cmd
# Create a test.bat file
@echo off
python scripts\run_tests.py
pause
```

---

## Test Coverage

### Coverage Goals

- **Overall**: Minimum 70% code coverage
- **Critical Modules**: Minimum 80% coverage for:
  - Data preprocessing
  - Model explainability
  - Evaluation metrics

### Viewing Coverage Reports

After running tests with coverage:

```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html

# Open the report
# Windows:
start htmlcov\index.html

# Linux/Mac:
open htmlcov/index.html
```

### Coverage by Module

| Module | Target Coverage | Description |
|--------|----------------|-------------|
| `src/data/preprocessor.py` | 80% | Data preprocessing pipeline |
| `src/models/explainer.py` | 75% | Model explainability |
| `src/utils/metrics.py` | 85% | Evaluation metrics |
| `src/utils/logger.py` | 60% | Logging utilities |

---

## Writing Tests

### Test Structure

```python
import pytest

@pytest.fixture
def sample_data():
    """Fixture providing test data"""
    return {"key": "value"}

class TestYourFeature:
    """Test suite for your feature"""
    
    def test_basic_functionality(self, sample_data):
        """Test basic functionality"""
        assert sample_data["key"] == "value"
    
    def test_edge_case(self):
        """Test edge case"""
        with pytest.raises(ValueError):
            # Code that should raise ValueError
            pass
```

### Best Practices

1. **Use Descriptive Names**
   ```python
   # Good
   def test_preprocessor_handles_missing_values_correctly()
   
   # Bad
   def test1()
   ```

2. **One Assertion Per Test** (when possible)
   ```python
   def test_accuracy_is_above_threshold():
       assert accuracy >= 0.85
   
   def test_precision_is_above_threshold():
       assert precision >= 0.80
   ```

3. **Use Fixtures for Setup**
   ```python
   @pytest.fixture
   def trained_model():
       model = XGBClassifier()
       model.fit(X_train, y_train)
       return model
   ```

4. **Test Edge Cases**
   ```python
   def test_empty_input():
       with pytest.raises(ValueError):
           preprocessor.transform(pd.DataFrame())
   
   def test_single_sample():
       result = preprocessor.transform(single_sample_df)
       assert result.shape[0] == 1
   ```

5. **Use Parametrize for Multiple Cases**
   ```python
   @pytest.mark.parametrize("threshold,expected", [
       (0.5, 0),
       (0.8, 1),
       (0.2, 0),
   ])
   def test_classification_with_threshold(threshold, expected):
       result = classify(0.7, threshold)
       assert result == expected
   ```

### Test Markers

Use markers to organize tests:

```python
@pytest.mark.unit
def test_unit_function():
    pass

@pytest.mark.integration
def test_integration_flow():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.model
def test_requires_trained_model():
    pass
```

Run specific markers:
```bash
pytest -m unit          # Run only unit tests
pytest -m "not slow"    # Skip slow tests
pytest -m integration   # Run only integration tests
```

---

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Push to main/master branch
- Pull requests
- Scheduled daily runs

See `.github/workflows/ci.yml` for configuration.

### Pre-commit Hooks

Run tests before committing:

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to test file
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
```

**2. Fixture Not Found**
```python
# Ensure fixture is in same file or conftest.py
# conftest.py fixtures are automatically discovered
```

**3. Tests Pass Locally But Fail in CI**
- Check Python version compatibility
- Verify all dependencies are in requirements.txt
- Ensure test data is not hard-coded to local paths

**4. Slow Tests**
```bash
# Run only fast tests
pytest -m "not slow"

# Use pytest-xdist for parallel execution
pip install pytest-xdist
pytest -n auto  # Use all CPU cores
```

---

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain minimum coverage thresholds
4. Add test documentation for complex cases
5. Update this README if adding new test files

---

**Last Updated:** August 2026
**Test Suite Version:** 1.0.0
