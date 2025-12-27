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
- **Logarithmic**: Compresses heavy-tailed distributions, useful for data spanning multiple orders of magnitude (prices, populations, word frequencies)
- **Square Root**: Stabilizes variance in Poisson-distributed data (count data, event frequencies, patient measurements)
- **Square**: Captures non-linear quadratic relationships (age effects, distance calculations)
- **Cube**: Captures strong non-linear relationships (accelerating growth patterns, S-curves)

#### Pairwise Transformations
- **Multiplication**: Creates interaction features (area from length × width, population from area × density)
- **Division**: Creates per-unit measures (BMI, price efficiency, density metrics)

### Additional Features
- Configurable number of results (`max_results`)
- Time limit option (`max_minutes`)
- Automatic filtering of invalid transformations (NaN, infinity)
- Input validation with helpful error messages

## Installation

```bash
pip install pandas numpy scikit-learn
```

## Quick Start

```python
import pandas as pd
from edu_autofe import EduAutoFE

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

## Usage

### Basic Usage

```python
from edu_autofe import EduAutoFE

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

## Tested Datasets

EduAutoFE has been tested on the following datasets:
- Titanic (Kaggle) - Binary classification
- Spaceship Titanic (Kaggle) - Binary classification
- Medical Cost Personal (Kaggle) - Regression
- California Housing (Scikit-learn) - Regression

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

## Next Steps

After using EduAutoFE, consider:
1. Applying recommended transformations to your data
2. Combining multiple good transformations together
3. Testing with other models (Random Forest, XGBoost, etc.)
4. Remember: these transformations help most with linear models

## License

MIT License - see [LICENSE](LICENSE) file for details.
