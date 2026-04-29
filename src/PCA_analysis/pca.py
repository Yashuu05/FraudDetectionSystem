# performing PCA on the processed dataset
import os
import sys
# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)
from src.utils import data_utils
from src.pipelines import model_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Project root = parent of src/
PROJECT_ROOT = root

def perform_pca(data):
    
    # list of features
    cat_cols = []
    num_cols = []

    # separate data into X and y
    df = data.copy()
    X = df.drop(["nameOrig","nameDest","isFraud","isFlaggedFraud"], axis=1)
    y = df["isFraud"]

    # identify categorical and numerical features
    for cols in X.columns:
        if X[cols].dtypes == "object":
            cat_cols.append(cols)
        else:
            num_cols.append(cols)
            
    # apply preprocessing on dataset
    preprocessor = model_pipeline.build_pipeline(cat_cols=cat_cols, num_cols=num_cols)
    X_preprocessed = preprocessor.fit_transform(X)
    # perform PCA
    pca = PCA(n_components=13)
    X_pca = pca.fit_transform(X_preprocessed)

    return X_pca, X_preprocessed, y
    

if __name__ == "__main__":
    
    # load a sample from the large dataset (full file is ~1.1 GB)
    dataset_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")
    print("Loading dataset sample (this may take a minute while streaming the file)...")
    dataset = data_utils.load_dataset_sample(
        file_path=dataset_path,
        sample_size=100_000,   # adjust upward if memory allows
        random_state=42,
        chunksize=50_000,
    )
    
    # perform PCA
    if dataset is not None:
        x_pca, x_preprocessed, y = perform_pca(data=dataset)
        
        # plot the graph
        plt.figure(figsize=(12, 5))
        y_numeric = pd.factorize(y)[0]
        plt.subplot(1, 2, 1)
        plt.scatter(x_preprocessed[:, 0], x_preprocessed[:, 1], c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
        plt.xlabel('Original Feature 1')
        plt.ylabel('Original Feature 2')
        plt.title('Before PCA: Using First 2 Standardized Features')
        plt.colorbar(label='Target classes')

        plt.subplot(1, 2, 2)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('After PCA')

        # save plot
        assets_dir = os.path.join(PROJECT_ROOT, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        plt.savefig(os.path.join(assets_dir, "pca_new.png"))
    
    else:
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")