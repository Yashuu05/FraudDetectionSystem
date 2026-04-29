from sklearn.inspection import permutation_importance
import mlflow
from sklearn.ensemble import RandomForestClassifier
from src.pipelines import model_pipeline
from src.utils import data_utils
from sklearn.pipeline import Pipeline
import os
import matplotlib.pyplot as plt
import pandas as pd

mlflow.set_experiment("fraud detection permuatation importance")
mlflow.autolog()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def perform_permuatation_importance(X_train, y_train):
    num_cols, cat_cols = data_utils.separate_cols_type(data=X_train)
    
    # build model
    model = RandomForestClassifier(n_jobs=-1, verbose=2, n_estimators=200, criterion='gini')
    preprocessor = model_pipeline.build_pipeline(cat_cols=cat_cols, num_cols=num_cols)
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ('model', model)
    ])
    print("\nfitting model...")
    with mlflow.start_run():
        rf_pipeline.fit(X_train, y_train)

    # Perform permutation importance
    print("\nimplementing Permutation Importance")
    # Get importance
    importance = rf_pipeline.feature_importances_
    # Summarize feature importance
    for i, v in enumerate(importance):
        print(f'Feature: {i}, Score: {v:.5f}')
        # save into the dictionary
        permutation_imp = {f"feature: {i}, Score: {v:.5f}"}

    # Plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    assets_dir = os.path.join(PROJECT_ROOT, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    plt.savefig(os.path.join(assets_dir, "permuration_imp.png"))

    return pd.DataFrame(permutation_imp)


if __name__ == "__main__":

    # load dataset
    dataset_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")
    print("Loading dataset sample (this may take a minute while streaming the file)...")
    dataset = data_utils.load_dataset_sample(
        file_path=dataset_path,
        sample_size=100_000,   # adjust upward if memory allows
        random_state=42,
        chunksize=50_000,
    )

    if dataset is None:
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    else:
        print("\ndataset loaded succesfully.")
        df = dataset.copy()
        X = df.drop(["nameOrig","nameDest","isFraud","isFlaggedFraud"], axis=1)
        y = df["isFraud"]
        X_train, X_test, y_train, y_test = data_utils.split_dataset(randomState=42, testSize=0.20, X=X, y=y)
