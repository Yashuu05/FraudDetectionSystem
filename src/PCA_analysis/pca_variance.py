import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

from src.utils import data_utils
from src.pipelines import model_pipeline

PROJECT_ROOT = root

def analyze_pca_variance(data):
    # Separate data into X and y
    df = data.copy()
    X = df.drop(["nameOrig", "nameDest", "isFraud", "isFlaggedFraud"], axis=1)
    
    # Identify categorical and numerical features
    cat_cols = [col for col in X.columns if X[col].dtype == "object"]
    num_cols = [col for col in X.columns if X[col].dtype != "object"]
            
    # Apply preprocessing
    print("applying preprocssing")
    preprocessor = model_pipeline.build_pipeline(cat_cols=cat_cols, num_cols=num_cols)
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Fit PCA with all components
    pca = PCA()
    print("fitting pca....")
    pca.fit(X_preprocessed)
    
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    
    return exp_var_pca, cum_sum_eigenvalues

if __name__ == "__main__":
    dataset_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")
    print("Loading dataset sample...")
    dataset = data_utils.load_dataset_sample(
        file_path=dataset_path,
        sample_size=100_000,
        random_state=42
    )
    
    if dataset is not None:
        exp_var, cum_var = analyze_pca_variance(dataset)
        
        # Print results
        print("\nIndividual Explained Variance Ratio:")
        for i, var in enumerate(exp_var):
            print(f"PC{i+1}: {var:.4f}")
            
        print("\nCumulative Explained Variance:")
        for i, cum in enumerate(cum_var):
            print(f"PCs 1-{i+1}: {cum:.4f}")

        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Individual variance (bar)
        plt.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.5, align='center',
                label='Individual explained variance')
        
        # Cumulative variance (step)
        plt.step(range(1, len(cum_var) + 1), cum_var, where='mid',
                 label='Cumulative explained variance', color='red')
        
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.title('Explained Variance Ratio for PCA Components')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save plot
        assets_dir = os.path.join(PROJECT_ROOT, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        plt.savefig(os.path.join(assets_dir, "pca_variance.png"))
        print(f"\nPlot saved to assets/pca_variance.png")

    else:
        print("Dataset not found.")
