import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from task_detector import get_model_and_scoring


def evaluate_feature(X_base, new_feature, y, task_type):
    """
    Evaluates a new feature by adding it to the base features and measuring performance.
    
    Args:
        X_base: Base feature dataframe
        new_feature: New feature dataframe to evaluate
        y: Target series
        task_type: "classification" or "regression"
        
    Returns:
        tuple: (mean_score, std_score)
    """
    scaler = StandardScaler()
    model, scoring, _ = get_model_and_scoring(task_type)
    
    # Concatenate new feature with existing ones
    X_with_new = pd.concat([X_base, new_feature], axis=1)
    
    # Scale and run cross validation
    X_scaled = scaler.fit_transform(X_with_new)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring=scoring)
    
    return scores.mean(), scores.std()
