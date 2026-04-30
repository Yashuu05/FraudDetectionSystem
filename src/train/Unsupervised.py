import os
import sys
import mlflow
import traceback

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

from src.utils import data_utils
from src.utils import model_utils
from src.pipelines import full_pipeline

PROJECT_ROOT = root
mlflow.set_experiment("fraud detection - unsupervised")
mlflow.autolog()

def train_unsupervised():
    print("========== Initialized Unsupervised Training (Isolation Forest) =========== ")
    
    # Step 1: Load dataset
    print("\n1. Loading dataset...")
    data_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")
    dataset = data_utils.load_dataset_sample(
        file_path=data_path,
        sample_size=100000,
        random_state=42,
        chunksize=50000
    )

    if dataset is None:
        raise FileNotFoundError(f"Error! Dataset {data_path} could not be loaded.")
        
    try:
        df = dataset.copy()
        
        # Step 2: Split dataset
        print("\n2. Splitting dataset...")
        # Dropping columns not used for features
        X = df.drop(["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"], axis=1)
        y = df['isFraud']
        
        X_train, X_test, y_train, y_test = data_utils.split_dataset(
            randomState=42,
            testSize=0.20,
            X=X,
            y=y
        )

        # Step 3: Loading model pipeline
        print("\n3. Building Isolation Forest pipeline...")
        categorical_cols, numerical_cols = data_utils.separate_cols_type(data=X)
        
        # Build the grid search for isolation forest
        grid_search = full_pipeline.build_full_pipeline(
            cat_cols=categorical_cols, 
            num_cols=numerical_cols,
            model_name="isoforest"
        )

        # Step 4: Fitting model
        print("\n4. Fitting model (Hyperparameter tuning)...")
        # Note: Isolation Forest is unsupervised, but we pass y_train for the GridSearch scoring (Recall)
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters found: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        
        # Step 5: Predicting and Evaluating
        print("\n5. Predicting on test set...")
        y_pred = best_model.predict(X_test)

        print("\n6. Evaluating model performance...")
        metrics_names = ["Recall", "Precision", "F1 Score", "Accuracy", "ROC AUC"]
        results = model_utils.evalulate_model(y_test=y_test, y_pred=y_pred)
        
        # results is [recall, precision, f1, acc, roc, report]
        for name, val in zip(metrics_names, results[:5]):
            print(f"{name} : {val:.4f}")
        
        print("\nClassification Report:")
        print(results[5])

        # Save the best model
        model_save_path = os.path.join(PROJECT_ROOT, "models", "isolation_forest_best.joblib")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model_utils.save_model(model_save_path, best_model)
        print(f"\nModel saved successfully at: {model_save_path}")

        print("\n============ Unsupervised training finished ============")

    except Exception as e:
        print(f"Error during unsupervised training: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    train_unsupervised()
