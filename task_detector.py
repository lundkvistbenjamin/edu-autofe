from sklearn.linear_model import LinearRegression, LogisticRegression


def detect_task_type(y):
    """
    Detects whether the task is classification or regression based on target variable.
    
    Args:
        y: Target series
        
    Returns:
        str: "classification" or "regression"
    """
    n_unique = y.nunique()
    
    if n_unique == 2:
        # Binary classification
        print(f"Detected: Binary Classification ({n_unique} classes)")
        print(f"   Model: Logistic Regression")
        print(f"   Why: Good for learning how features affect binary outcomes\n")
        return "classification"
    else:
        # Regression
        print(f"Detected: Regression ({n_unique} unique values)")
        print(f"   Model: Linear Regression")
        print(f"   Why: Good for learning how features affect continuous outcomes\n")
        return "regression"


def get_model_and_scoring(task_type):
    """
    Returns the appropriate model and scoring metric for the task type.
    
    Args:
        task_type: "classification" or "regression"
        
    Returns:
        tuple: (model, scoring_metric, metric_name)
    """
    if task_type == "regression":
        return LinearRegression(), "r2", "R2"
    else:
        return LogisticRegression(max_iter=1000, random_state=42), "accuracy", "Accuracy"
