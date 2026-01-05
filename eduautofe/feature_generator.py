import pandas as pd
import numpy as np


def generate_candidates(X):
    """
    Generates candidate feature transformations from the input data.
    
    Args:
        X: Feature dataframe
        
    Returns:
        tuple: (candidates_list, n_continuous_cols)
            candidates_list: List of (feature_name, feature_series, description) tuples
            n_continuous_cols: Number of continuous columns in X
    """
    candidates = []
    
    # Single column transformations with descriptions
    transforms = {
        "log": (lambda x: np.log(x + 1), "Log transformation of {col}. Compresses the long tail in the high part of heavy-tailed distributions and expands the low part, making data more normally distributed. \nCommon applications: data that spreads over several orders of magnitude, such as prices, populations, incomes, number of reviews, word frequencies, and sales figures."),
        "sqrt": (lambda x: np.sqrt(np.abs(x)), "Square root of {col}. Used for Poisson-distributed data where variance equals the mean. Stabilizes variance so it is no longer dependent on the mean. Also used for compressing the long tail and strengthening the signal. \nCommon applications: count data, event frequencies, patient measurements with extreme values (like weight or blood pressure), and visit counts."),
        "square": (lambda x: x ** 2, "Square of {col}. Polynomial transformation used to capture non-linear patterns in data, which is especially valuable for linear models that have difficulty finding these relationships on their own. Adds higher-order components to create new, more complex features. \nCommon applications: variables with quadratic relationships, such as age effects, distance calculations, and diminishing returns patterns."),
        "cube": (lambda x: x ** 3, "Cube of {col}. Polynomial transformation used to capture non-linear patterns in data, which is especially valuable for linear models that have difficulty finding these relationships on their own. Adds higher-order components to create new, more complex features. \nCommon applications: variables with strong non-linear relationships, such as accelerating growth patterns, S-shaped curves, and compound effects."),
    }

    # Generate single column transformations
    for col in X.columns:
        for name, (func, desc_template) in transforms.items():
            try:
                transformed = func(X[col])
                
                # Skip if we get NaN or infinity
                if pd.isna(transformed).any() or np.isinf(transformed).any():
                    continue
                
                feature_name = f"{name}({col})"
                feature_series = pd.Series(transformed, index=X.index, name=feature_name)
                description = desc_template.format(col=col)
                
                candidates.append((feature_name, feature_series, description))
            except:
                continue
    
    # Pairwise operations with descriptions
    operations = {
        "multiply": (lambda a, b: a * b, "Multiplication of {col1} and {col2}. Captures interaction effects and combined impact. \nCommon applications: creating area features (length × width), calculating population (area × density), computing total cost (price × quantity), and modeling combined effects."),
        "divide": (lambda a, b: a / (b + 1e-5), "Division of {col1} by {col2}. Used to create per-unit measures by dividing one variable by another. \nCommon applications: calculating BMI (weight ÷ height²), price efficiency (price ÷ area), density metrics (population ÷ area), and normalized rates."),
    }
    
    # Separate continuous and binary columns
    continuous_cols = [col for col in X.columns if X[col].nunique() > 2]
    binary_cols = [col for col in X.columns if X[col].nunique() == 2]

    # Generate pairwise transformations for continuous × continuous
    for i, col1 in enumerate(continuous_cols):
        for col2 in continuous_cols[i+1:]:
            for op_name, (func, desc_template) in operations.items():
                try:
                    transformed = func(X[col1], X[col2])
                    
                    # Skip invalid values
                    if pd.isna(transformed).any() or np.isinf(transformed).any():
                        continue
                    
                    feature_name = f"{col1} {op_name} {col2}"
                    feature_series = pd.Series(transformed, index=X.index, name=feature_name)
                    description = desc_template.format(col1=col1, col2=col2)
                    
                    candidates.append((feature_name, feature_series, description))
                except:
                    continue
    
    # Generate pairwise transformations for continuous × binary
    for cont_col in continuous_cols:
        for bin_col in binary_cols:
            try:
                transformed = X[cont_col] * X[bin_col]
                
                # Skip invalid values
                if pd.isna(transformed).any() or np.isinf(transformed).any():
                    continue
                
                feature_name = f"{cont_col} multiply {bin_col}"
                feature_series = pd.Series(transformed, index=X.index, name=feature_name)
                description = f"Multiplication of {cont_col} and {bin_col}. Captures interaction effects and combined impact. \nCommon applications: creating area features (length × width), calculating population (area × density), computing total cost (price × quantity), and modeling combined effects."
                
                candidates.append((feature_name, feature_series, description))
            except:
                continue
    
    return candidates, len(continuous_cols)
