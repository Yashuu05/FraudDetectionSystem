import os
import sys

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

from src.utils import data_utils
from src.utils import model_utils
from src.pipelines import full_pipeline
import mlflow

PROJECT_ROOT = root
mlflow.set_experiment("fraud detection")
mlflow.autolog()

# step 1 load dataset
MODEL_TO_TRAIN = "xgb"  # Options: "xgb", "isoforest"

print("========== Initialized training =========== ")
print(f"\nModel selected: {MODEL_TO_TRAIN}")
print("\n1. loading dataset...")
data_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")
dataset = data_utils.load_dataset_sample(
    file_path=data_path,
    sample_size=100000,
    random_state=42,
    chunksize=50000
)

if dataset is None:
    raise FileNotFoundError(f"Error! dataset {data_path} couldn't load.")
    
else:

    try:
        df = dataset.copy()
        print("\n2. Splitting dataset...")

        X = df.drop(["isFraud", "isFlaggedFraud","nameOrig","nameDest"], axis=1)
        y = df['isFraud']
        X_train, X_test, y_train, y_test = data_utils.split_dataset(
            randomState=42,
            testSize=0.20,
            X=X,
            y=y
        )

        print("\n3. loading model pipeline...")
        categorical_cols, numerical_cols = data_utils.separate_cols_type(data=X)
        grid_search = full_pipeline.build_full_pipeline(
            cat_cols=categorical_cols, 
            num_cols=numerical_cols,
            model_name=MODEL_TO_TRAIN
        )

        print(f"\n4. fitting model ({MODEL_TO_TRAIN})...")
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)

        print ("\n5. Evaluating model...")
        metrics_names = ["Recall", "Precision", "F1 Score", "Accuracy", "ROC AUC"]
        results = model_utils.evalulate_model(y_test=y_test, y_pred=y_pred)
        
        # results is [recall, precision, f1, acc, roc, report]
        for i, (name, val) in enumerate(zip(metrics_names, results[:5])):
            print(f"{name} : {val:.4f}")
        
        print("\nClassification Report:")
        print(results[5])

        print("============ training finished ============")

    except Exception as e:
        import traceback
        print(f"Error during training: {str(e)}")
        traceback.print_exc()