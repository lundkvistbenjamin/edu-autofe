import pandas as pd
import numpy as np
import random
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from validators import validate_input
from task_detector import detect_task_type, get_model_and_scoring
from feature_generator import generate_candidates
from evaluator import evaluate_feature
from results_printer import print_results


class EduAutoFE:
    
    def __init__(self, max_results=5, max_minutes=None):
        self.max_results = max_results
        self.max_minutes = max_minutes
        self.results = []
    
    def fit(self, X, y):
        # Validate input
        X, y = validate_input(X, y)
        
        # Set task type
        self.task_type = detect_task_type(y)
        
        # Calculate baseline
        print("Calculating baseline performance...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model, scoring, metric = get_model_and_scoring(self.task_type)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring=scoring)
        baseline_score = scores.mean()
        baseline_std = scores.std()
        
        # Generate all candidates
        print(f"\nGenerating candidate transformations...")
        candidates, n_continuous = generate_candidates(X)
        
        # Randomize order so time limited runs test different candidates each time
        random.shuffle(candidates)

        # Print the amount of candidates created
        print(f"   Generated {len(candidates)} candidates")
        
        # Print which mode is used
        if self.max_minutes:
            print(f"\nEvaluating candidates (max {self.max_minutes} minutes)...")
        else:
            print(f"\nEvaluating candidates (exhaustive search)...")
        print(f"   This may take a moment...\n")
        
        # Track time
        start_time = time.time()
        candidates_tested = 0
        
        # Evaluate each candidate
        for feature_name, feature_series, description in candidates:
            # Check time limit
            if self.max_minutes:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes >= self.max_minutes:
                    print(f"   Time limit reached ({self.max_minutes} min)")
                    print(f"   Tested {candidates_tested} of {len(candidates)} candidates\n")
                    break
            
            try:
                # Wrap in dataframe for concat
                new_feature_df = pd.DataFrame({feature_name: feature_series})
                
                score, std = evaluate_feature(X, new_feature_df, y, self.task_type)
                improvement = score - baseline_score
                
                candidates_tested += 1
                
                # Only keep if it improves the score
                if improvement > 0:
                    self.results.append({
                        "feature": feature_name,
                        "score": score,
                        "std": std,
                        "improvement": improvement,
                        "description": description
                    })
            except:
                continue
        
        # Check if there were no features that improved the performance
        if not self.results:
            print("No improvements found over baseline.")
            return pd.DataFrame()
        
        # Sort results by improvement
        self.results.sort(key=lambda x: x["improvement"], reverse=True)
        results_df = pd.DataFrame(self.results[:self.max_results])
        
        # Print the results
        print_results(results_df, baseline_score, baseline_std, self.task_type)
        
        return results_df
