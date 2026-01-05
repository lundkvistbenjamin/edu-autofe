def print_results(df, baseline_score, baseline_std, task_type):
    """
    Prints the results of feature engineering in a formatted way.
    
    Args:
        df: Results dataframe with columns: feature, score, std, improvement, description
        baseline_score: Baseline model score
        baseline_std: Baseline model standard deviation
        task_type: "classification" or "regression"
    """
    metric = "Accuracy" if task_type == "classification" else "R2"
    
    print("\n" + "="*60)
    print(f"{'RESULTS':^60}")
    print("="*60)
    print(f"\nBaseline {metric}: {baseline_score:.4f} +/- {baseline_std:.4f}")
    
    if len(df) > 0:
        best = df.iloc[0]["improvement"]
        print(f"Best improvement: +{best:.4f}")
    
    print(f"\nTop {len(df)} Transformations:")
    print("-"*60)
    
    for i, row in df.iterrows():
        print(f"\n{i+1}. {row['feature']}")
        print(f"{metric}: {row['score']:.4f} +/- {row['std']:.4f} (+{row['improvement']:.4f})")
        print(f"{row['description']}")
    
    print("\n" + "="*60)
    
    print("\nWhat does this mean?")
    if task_type == "classification":
        print("Accuracy shows what % of predictions are correct.")
        print("Higher accuracy = better model performance.")
        print("+/- std shows consistency across different data splits.")
    else:
        print("R2 shows what % of variation in the target the model explains.")
        print("R2 = 0.8 means the model explains 80% of the pattern.")
        print("+/- std shows consistency across different data splits.")
    
    print("\nNext steps:")
    print("1. Try applying these transformations to your data")
    print("2. Combine multiple good transformations together")
    print("3. Test with other models (Random Forest, XGBoost, etc.)")
    print("4. Remember: these transformations help most with linear models!")
    print("\n" + "="*60)