import pandas as pd
import numpy as np

def validate_input(X, y):
    """
    Validates input data X and y for common issues.
    
    Args:
        X: Feature dataframe
        y: Target series
        
    Returns:
        Validated X and y
        
    Raises:
        TypeError: If X is not a DataFrame or y is not a Series
        ValueError: If data has issues (too many columns, ID columns, text columns, missing values)
    """
    
    # X needs to be a dataframe
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Error: X must be a pandas DataFrame\n\n"
            "Fix: X = pd.DataFrame(X)"
        )
    
    # Y needs to be a series
    if not isinstance(y, pd.Series):
        raise TypeError(
            "Error: y must be a pandas Series\n\n"
            'Fix: y = pd.Series(y) or y = df["target_column"]'
        )
    
    # Check for too many columns
    if len(X.columns) > 50:
        prefix_counts = {}
        for col in X.columns:
            if '_' in col:
                prefix = col.split('_')[0]
                if prefix not in prefix_counts:
                    prefix_counts[prefix] = []
                prefix_counts[prefix].append(col)
        
        high_card_prefixes = {prefix: cols for prefix, cols in prefix_counts.items() if len(cols) > 10}
        
        if high_card_prefixes:
            sorted_culprits = sorted(high_card_prefixes.items(), key=lambda x: len(x[1]), reverse=True)
            
            culprits_text = ""
            for prefix, cols in sorted_culprits:
                culprits_text += f"   '{prefix}' columns ({len(cols)} columns)\n"
            
            raise ValueError(
                f"Error: Too many columns ({len(X.columns)})\n\n"
                f"Likely culprits:\n{culprits_text}\n"
                f'Drop with: X = X.drop(columns=["Col1", "Col2", ...])'
            )
        else:
            raise ValueError(
                f"Error: Too many columns ({len(X.columns)})\n\n"
                f"This happens when your dataset has too many features\n"
                f"or one-hot encoding created too many columns.\n\n"
                f'Drop with: X = X.drop(columns=["col1", "col2", ...])'
            )
    
    # Check for ID columns
    id_patterns = ["id", "index", "key", "number", "code"]
    for col in X.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in id_patterns):
            if X[col].dtype in [np.int64, np.float64]:
                uniqueness = X[col].nunique() / len(X)
                if uniqueness > 0.95:
                    raise ValueError(
                        f"Error: '{col}' looks like an ID column\n\n"
                        f"Drop with: X = X.drop(columns=['{col}'])"
                    )
    
    # Check for text columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        raise ValueError(
            f"Error: Found text columns: {cat_cols}\n"
            f"This can either be addressed by label encoding them if they should remain ordered in a single column,\nor one-hot encoding them if they should be unordered.\n"
            f"If they add no value they can be dropped completely.\n\n"
            f"One-hot encode with: X = pd.get_dummies(X, drop_first=True, dtype=int)\n"
            f'Drop with: X = X.drop(columns=["text_col"])\n'
            f'Label encode with: X["col"] = LabelEncoder().fit_transform(X["col"])'
        )
    
    # Check for missing values in X
    if X.isnull().any().any():
        cols_with_missing = X.columns[X.isnull().any()].tolist()
        raise ValueError(
            f"Error: Missing values in: {cols_with_missing}\n\n"
            f"Fill with median: X = X.fillna(X.median())\n"
            f"Fill with mean: X = X.fillna(X.mean())\n"
            f'Drop with: X = X.drop(columns=["col_with_missing"])'
        )
    
    # Check for missing values in y
    if y.isnull().any():
        raise ValueError(
            f"Error: {y.isnull().sum()} missing values in y\n\n"
            f"Fill with: y = y.fillna(y.median())"
        )
    
    return X, y
