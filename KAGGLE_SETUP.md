# EduAutoFE - Setup for Kaggle

## Quick Start in Kaggle

### Option 1: Direct File Upload
1. Upload all `.py` files to your Kaggle notebook:
   - `eduautofe.py`
   - `validators.py`
   - `task_detector.py`
   - `feature_generator.py`
   - `evaluator.py`
   - `results_printer.py`
   - `__init__.py`

2. In your first code cell, run:
```python
!pip install -q pandas numpy scikit-learn
```

3. Import and use:
```python
from eduautofe import EduAutoFE

# Your code here
autofe = EduAutoFE(max_results=5, max_minutes=2)
results = autofe.fit(X, y)
```

### Option 2: Install from Requirements
1. Upload `requirements.txt` along with all `.py` files
2. Run in first cell:
```python
!pip install -r requirements.txt -q
```

### Option 3: GitHub Integration (if you upload to GitHub)
```python
!pip install git+https://github.com/YOUR_USERNAME/eduautofe.git
```

## Basic Usage

```python
import pandas as pd
from eduautofe import EduAutoFE

# Prepare your data
X = pd.DataFrame(...)  # Your features
y = pd.Series(...)      # Your target

# Run automated feature engineering
autofe = EduAutoFE(max_results=5, max_minutes=2)
results = autofe.fit(X, y)

# Results dataframe contains:
# - feature: name of the transformation
# - score: cross-validation score with this feature
# - std: standard deviation of the score
# - improvement: improvement over baseline
# - description: explanation of what the feature does
```

## Parameters

- `max_results` (default=5): Maximum number of top features to return
- `max_minutes` (default=None): Time limit in minutes. If None, runs exhaustive search.

## Requirements

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0

All of these are pre-installed in Kaggle environments.
