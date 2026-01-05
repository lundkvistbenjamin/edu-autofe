# EduAutoFE

EduAutoFE is an educational automated feature engineering tool designed to help beginners understand and apply fundamental feature engineering techniques. The tool automatically generates, evaluates, and explains feature transformations to improve machine learning model performance.

## Overview

EduAutoFE simplifies the feature engineering process by:
- Automatically generating candidate features through mathematical transformations
- Evaluating each transformation's impact on model performance
- Providing clear explanations of what each transformation does and when to use it
- Offering beginner-friendly error messages with suggested fixes

## Features

### Automatic Problem Detection
- Binary classification detection (2 unique values in target)
- Regression detection (continuous target values)
- Automatic model selection based on problem type

### Model Selection
- **Classification**: Logistic Regression with accuracy scoring
- **Regression**: Linear Regression with R² scoring
- 5-fold cross-validation for robust evaluation

### Output Format
- DataFrame containing top-performing transformations
- Baseline model performance metrics
- Performance improvement for each transformation
- Standard deviation showing consistency across data splits
- Educational descriptions for each transformation

### Transformation Types

#### Single-Variable Transformations
- **Logarithmic**
- **Square Root**
- **Square**
- **Cube**

#### Pairwise Transformations
- **Multiplication**
- **Division**

### Additional Features
- Configurable number of results (`max_results`)
- Time limit option (`max_minutes`)
- Automatic filtering of invalid transformations (NaN, infinity)
- Input validation with helpful error messages

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/lundkvistbenjamin/edu-autofe.git
```

### For Kaggle

In your Kaggle notebook:

```python
!pip install git+https://github.com/lundkvistbenjamin/edu-autofe.git
```

### Requirements (already included)

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0

## Quick Start

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from eduautofe import EduAutoFE

# Load your data
df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Prepare data
X = pd.get_dummies(X, drop_first=True, dtype=int)  # Encode categorical variables
X = X.fillna(X.median())  # Fill missing values
y = y.fillna(y.median())

# Run EduAutoFE
model = EduAutoFE(max_results=5)
results = model.fit(X, y)
```

## Complete Kaggle Example (Titanic)

```python
# Install from GitHub
!pip install git+https://github.com/lundkvistbenjamin/edu-autofe.git

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from eduautofe import EduAutoFE

# Load Titanic data
df = pd.read_csv("/kaggle/input/titanic/train.csv")
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Drop unnecessary columns
X = X.drop(columns=["PassengerId", "Name"])

# Encode categorical variables
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Fill missing values
X = X.fillna(X.median())
y = y.fillna(y.median())

# Run EduAutoFE
model = EduAutoFE(max_minutes=2)
results = model.fit(X, y)
```

## Usage

### Basic Usage

```python
from eduautofe import EduAutoFE

# Create instance with default settings
model = EduAutoFE()
results = model.fit(X, y)
```

### With Time Limit

```python
# Limit execution time to 5 minutes
model = EduAutoFE(max_minutes=5)
results = model.fit(X, y)
```

### Custom Number of Results

```python
# Show top 10 transformations
model = EduAutoFE(max_results=10)
results = model.fit(X, y)
```

## Input Requirements

### X (Features)
- Must be a pandas DataFrame
- All columns must be numeric
- No missing values
- No ID-like columns (columns with >95% unique values)
- Maximum 100 columns

### y (Target)
- Must be a pandas Series
- No missing values
- Binary values (0/1) for classification
- Continuous values for regression

## Data Preparation

### Handling Categorical Variables

```python
# One-hot encoding
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Or label encoding
from sklearn.preprocessing import LabelEncoder
X["column"] = LabelEncoder().fit_transform(X["column"])
```

### Handling Missing Values

```python
# Fill with median
X = X.fillna(X.median())
y = y.fillna(y.median())

# Or fill with mean
X = X.fillna(X.mean())
```

### Removing Unwanted Columns

```python
# Drop ID columns
X = X.drop(columns=["PassengerId"])

# Drop text columns
X = X.drop(columns=["Name", "Ticket"])
```

## Output Interpretation

### Classification Output
- **Accuracy**: Percentage of correct predictions
- **+/- std**: Consistency across different data splits
- Higher accuracy indicates better performance

### Regression Output
- **R²**: Percentage of variation explained by the model
- **+/- std**: Consistency across different data splits
- R² closer to 1 indicates better fit

## Example Output

```
============================================================
                          RESULTS                           
============================================================

Baseline Accuracy: 0.7901 +/- 0.0157
Best improvement: +0.0123

Top 5 Transformations:
------------------------------------------------------------

1. log(Age)
Accuracy: 0.8025 +/- 0.0274 (+0.0123)
Log transformation of Age. Compresses the long tail in the high part 
of heavy-tailed distributions and expands the low part, making data 
more normally distributed.
Common applications: data that spreads over several orders of 
magnitude, such as prices, populations, incomes, number of reviews, 
word frequencies, and sales figures.
```

## Limitations

- Designed for educational purposes, not production optimization
- Limited to basic mathematical transformations
- Evaluates transformations individually, not in combination
- Uses simple linear models (interpretability over performance)
- Maximum 100 features to avoid excessive computation time

## Technical Details

### Dependencies
- pandas
- numpy
- scikit-learn

### Evaluation Method
- StandardScaler for feature scaling
- 5-fold cross-validation
- Transformation candidates are randomized for time-limited runs

## Acknowledgments

This README was formatted with assistance from Claude AI (Anthropic).

## License

MIT License - see [LICENSE](LICENSE) file for details.
