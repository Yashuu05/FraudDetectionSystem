import json
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)
PROJECT_ROOT = root

from src.utils.model_utils import load_model
from src.feature_engineering import build_features
from app.system_1.script import generate_data

def load_constants():
    try:
        # Load mean and stdev for Z-score
        constants_path = os.path.join(PROJECT_ROOT, "app", "system_2", "constants.json")
        with open(constants_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        mean = data['mean']
        std = data["std"]

        # Load weights (w1, w2, w3) from best_weights.json
        weights_path = os.path.join(PROJECT_ROOT, "models", "best_weights.json")
        with open(weights_path, "r", encoding="utf-8") as f:
            w_data = json.load(f)
            
        w1 = w_data["best_weights"]["w1_xgb"]
        w2 = w_data["best_weights"]["w2_iso"]
        w3 = w_data["best_weights"]["w3_rule"]
        
        return mean, std, w1, w2, w3

    except Exception as e:
        print(f"Error loading constants: {str(e)}")
        return None

def load_all_models():
    xgb_path = os.path.join(PROJECT_ROOT, "models", "xgb_classifier_best.joblib")
    iso_path = os.path.join(PROJECT_ROOT, "models", "isolation_forest_best.joblib")
    
    xgb = load_model(file_path=xgb_path)
    iso_forest = load_model(file_path=iso_path)
    
    if xgb is None or iso_forest is None:
        raise FileNotFoundError("One or more model files not found in models/ directory.")
    
    print("Models loaded successfully!")
    return xgb, iso_forest

def calculate_risk_score(data_input, mean, stdev, w1, w2, w3, xgb_model, iso_model):
    # 1. Apply Feature Engineering (Required for the models)
    processed_data = build_features(data_input.copy())
    
    # Ensure correct columns for models (dropping same cols as in training)
    # The pipelines handle categorical encoding, but we need the engineered columns
    X = processed_data.copy()
    if 'nameOrig' in X.columns: X = X.drop('nameOrig', axis=1)
    if 'nameDest' in X.columns: X = X.drop('nameDest', axis=1)

    # 2. Calculate individual scores
    
    # Z-score Rule: 1 if |Z| > 3, else 0
    amount = float(X["amount"].iloc[0])
    z_val = (amount - mean) / stdev
    z_rule_score = 1 if abs(z_val) > 3 else 0

    # XGB Score: Probability for class 1
    xgb_score = float(xgb_model.predict_proba(X)[:, 1][0])

    # Isolation Forest Score: Already 0 or 1 from our pipeline wrapper
    iso_score = float(iso_model.predict(X)[0])

    # 3. Final Weighted Score
    risk_score = (w1 * xgb_score) + (w2 * iso_score) + (w3 * z_rule_score)
    
    return round(risk_score, 4), {
        "xgb": round(xgb_score, 4),
        "iso": iso_score,
        "z_rule": z_rule_score,
        "z_val": round(z_val, 2)
    }

def run(num):
    # Load models 
    print("Loading resources...")
    try:
        xgb, iso = load_all_models()
        constants = load_constants()
        if constants is None: return
        mean, std, w1, w2, w3 = constants
    except Exception as e:
        print(f"Setup failed: {e}")
        return

    print(f"\nWeights used: XGB={w1}, IsoForest={w2}, Z-Rule={w3}")
    print("-" * 50)

    # Calculate risk score for each transaction data
    for i in range(num):
        data_input, display_data = generate_data()
        
        score, details = calculate_risk_score(
            data_input=data_input, 
            mean=mean, 
            stdev=std, 
            w1=w1, w2=w2, w3=w3,
            xgb_model=xgb,
            iso_model=iso
        )
        
        print(f"Transaction ID: {display_data['transaction_id']}")
        print(f"Type: {display_data['type']} | Amount: ${display_data['amount']:,.2f}")
        print(f"Scores -> XGB: {details['xgb']}, Iso: {details['iso']}, Z-Rule: {details['z_rule']} (Z={details['z_val']})")
        print(f"FINAL RISK SCORE: {score}")
        
        # Risk level logic
        if score > 0.6:
            print("STATUS: [CRITICAL] High Fraud Risk Detected!")
        elif score > 0.4:
            print("STATUS: [WARNING] Medium Risk - Review Recommended")
        else:
            print("STATUS: [SAFE] Low Risk")
        print("-" * 50)

if __name__ == "__main__":
    try:
        num_input = input("Enter number of transactions to simulate (default 5): ")
        num = int(num_input) if num_input.strip() else 5
        run(num=num)
    except ValueError:
        print("Please enter a valid number.")