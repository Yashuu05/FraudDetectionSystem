import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Add project root to path to ensure imports work correctly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import data_utils
from src.utils.model_utils import load_model

def perform_permutation_importance(X_test, y_test, model, n_repeats=5):
    """
    Computes permutation importance for a given model and test set.
    
    Args:
        X_test: Test features
        y_test: Test target
        model: Trained estimator or pipeline
        n_repeats: Number of times to permute each feature
        
    Returns:
        DataFrame containing importance mean and std for each feature.
    """
    print(f"Calculating permutation importance with {n_repeats} repeats...")
    # Permutation importance can be slow for large datasets, so we use a reasonable n_repeats
    result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring='recall' # Using recall as it's critical for fraud detection
    )
    
    # Create a DataFrame for better visualization and saving
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=False)
    
    return importance_df

def save_importance_plot(importance_df, output_path):
    """
    Plots the feature importance and saves it to a file.
    
    Args:
        importance_df: DataFrame with importance scores
        output_path: Path to save the plot image
    """
    plt.figure(figsize=(12, 10))
    # Select top 25 features for better readability if there are many
    top_n = 25
    plot_df = importance_df.head(top_n)
    
    # Using horizontal bar chart
    bars = plt.barh(plot_df['feature'], plot_df['importance_mean'], 
                    xerr=plot_df['importance_std'], align='center', 
                    color='skyblue', edgecolor='navy', alpha=0.8)
    
    plt.xlabel('Permutation Importance (Mean decrease in Recall)')
    plt.ylabel('Features')
    plt.title(f'Top {len(plot_df)} Features via Permutation Importance\n(Model: XGBoost Fraud Classifier)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Importance plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Define paths
    dataset_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")
    model_path = os.path.join(PROJECT_ROOT, "models", "xgb_classifier_best.joblib")
    results_csv_path = os.path.join(PROJECT_ROOT, "models", "permutation_importance_scores.csv")
    results_plot_path = os.path.join(PROJECT_ROOT, "assets", "permutation_importance.png")

    print("--- Permutation Importance Analysis ---")

    # 1. Load dataset sample
    print("\nStep 1: Loading dataset sample...")
    # Using a sample size that is representative but manageable for permutation importance
    dataset = data_utils.load_dataset_sample(
        file_path=dataset_path,
        sample_size=30_000, 
        random_state=42,
        chunksize=50_000
    )

    if dataset is None:
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    # 2. Load model
    print("\nStep 2: Loading trained model...")
    model = load_model(file_path=model_path)
    if model is None:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # 3. Data Preparation
    print("\nStep 3: Preparing data for importance calculation...")
    # Drop columns not used in training (consistent with src/train/Supervised.py)
    drop_cols = ["nameOrig", "nameDest", "isFraud", "isFlaggedFraud"]
    X = dataset.drop(columns=[col for col in drop_cols if col in dataset.columns], axis=1)
    y = dataset["isFraud"]

    # 4. Split data
    print("\nStep 4: Splitting dataset (using test set for analysis)...")
    _, X_test, _, y_test = data_utils.split_dataset(randomState=42, testSize=0.20, X=X, y=y)

    # 5. Perform Permutation Importance
    print("\nStep 5: Performing Permutation Importance...")
    # n_repeats=5 is a good balance between speed and stability for initial analysis
    importance_df = perform_permutation_importance(X_test, y_test, model, n_repeats=5)

    # 6. Save results to models folder
    print(f"\nStep 6: Saving importance scores to {results_csv_path}")
    importance_df.to_csv(results_csv_path, index=False)

    # 7. Generate and save plot to assets
    print("\nStep 7: Generating and saving importance plot...")
    save_importance_plot(importance_df, results_plot_path)

    print("\n" + "="*40)
    print("Permutation Importance analysis completed successfully.")
    print("="*40)
    print(f"Top 10 most influential features:\n{importance_df.head(10).to_string(index=False)}")
