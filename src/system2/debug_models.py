import os
import sys
import joblib as jb
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)
PROJECT_ROOT = root

from src.utils.model_utils import load_model
from src.utils.data_utils import load_dataset_sample, split_dataset

def debug():
    xgb_model_path = os.path.join(PROJECT_ROOT, "models", "xgb_classifier_best.joblib")
    iso_model_path = os.path.join(PROJECT_ROOT, "models", "isolation_forest_best.joblib")
    
    print(f"Loading XGB from: {xgb_model_path}")
    xgb_model = load_model(xgb_model_path)
    print(f"Loading IsoForest from: {iso_model_path}")
    iso_model = load_model(iso_model_path)
    
    for name, model in [("XGB", xgb_model), ("IsoForest", iso_model)]:
        print(f"\nChecking {name} model:")
        print(f"Type: {type(model)}")
        try:
            # Check if pipeline is fitted
            # For a pipeline, we check the last step usually, but check_is_fitted(model) works if it has been fitted
            check_is_fitted(model)
            print("Status: FITTED")
        except Exception as e:
            print(f"Status: NOT FITTED? Error: {e}")
            
    # Sample data
    data_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")
    df = load_dataset_sample(data_path, sample_size=1000)
    X = df.drop(["isFraud", "isFlaggedFraud", "nameDest", "nameOrig"], axis=1)
    
    print("\nTesting XGB predict_proba...")
    try:
        proba = xgb_model.predict_proba(X.head(5))
        print(f"XGB Proba sample:\n{proba}")
    except Exception as e:
        print(f"XGB Error: {e}")
        
    print("\nTesting IsoForest predict...")
    try:
        preds = iso_model.predict(X.head(5))
        print(f"IsoForest Preds sample: {preds}")
    except Exception as e:
        print(f"IsoForest Error: {e}")

if __name__ == "__main__":
    debug()
