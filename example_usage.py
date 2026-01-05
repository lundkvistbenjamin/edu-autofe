"""
Example usage of EduAutoFE

This script shows how to use the EduAutoFE class for automated feature engineering.
"""

import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from eduautofe import EduAutoFE

# Example 1: Regression task
print("=" * 60)
print("EXAMPLE 1: REGRESSION TASK")
print("=" * 60)

# Load diabetes dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Run automated feature engineering (limited to 1 minute for demo)
autofe = EduAutoFE(max_results=5, max_minutes=1)
results = autofe.fit(X, y)

print("\n\n")

# Example 2: Classification task
print("=" * 60)
print("EXAMPLE 2: CLASSIFICATION TASK")
print("=" * 60)

# Load breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Run automated feature engineering (exhaustive search for small dataset)
autofe = EduAutoFE(max_results=3)
results = autofe.fit(X, y)
