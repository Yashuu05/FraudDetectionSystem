import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import recall_score, precision_score, f1_score

warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)
PROJECT_ROOT = root

from src.utils.model_utils import load_model
from src.utils.data_utils import load_dataset_sample, split_dataset

def load_resources():
    print("1. Loading models...")
    xgb_model_path = os.path.join(PROJECT_ROOT, "models", "xgb_classifier_best.joblib")
    iso_model_path = os.path.join(PROJECT_ROOT, "models", "isolation_forest_best.joblib")
    
    xgb_model = load_model(file_path=xgb_model_path)
    iso_model = load_model(file_path=iso_model_path)

    if xgb_model is None or iso_model is None:
        raise FileNotFoundError("Could not load one or more models. Please check models/ directory.")

    # load datasets
    print("2. Loading dataset...")
    data_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")
    dataset = load_dataset_sample(
        file_path=data_path,
        sample_size=100000,
        random_state=42,
        chunksize=100000
    )

    if dataset is None:
        raise FileNotFoundError(f"Could not load dataset from {data_path}")

    return xgb_model, iso_model, dataset

def calculate_z_score_rule(df):
    """
    Implements the Z-score rule based on the 'amount' column.
    If Z-score > 3 or Z-score < -3, score = 1, else 0.
    """
    amounts = df['amount']
    z_scores = stats.zscore(amounts)
    rule_scores = ((z_scores > 3) | (z_scores < -3)).astype(int)
    return rule_scores

if __name__ == "__main__":
    print("DEBUG: Starting script execution...")
    print("========== Initialized Best Weights Search =========== ")
    
    # 1: load resources
    xgb_model, iso_model, df = load_resources()

    # 2: split dataset (using same logic as training)
    print("3. Splitting dataset for validation...")
    X = df.drop(["isFraud", "isFlaggedFraud", "nameDest", "nameOrig"], axis=1)
    y = df['isFraud']
    _, X_test, _, y_test = split_dataset(randomState=42, testSize=0.20, X=X, y=y)

    print(f"Validation set size: {len(X_test)}")

    # 3. Calculate individual scores on X_test once
    print("4. Calculating individual scores...")
    
    # XGB Scores (probabilities for class 1)
    print("   - Calculating XGB scores...")
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
    
    # Isolation Forest Scores
    print("   - Calculating Isolation Forest scores...")
    iso_preds = iso_model.predict(X_test)
    # The wrapper already returns 1 for fraud (anomaly) and 0 for legit.
    # So we don't need the (1 - iso_preds) / 2 conversion.
    iso_scores = iso_preds.astype(float)
    
    # Z-Score Rule Scores
    print("   - Calculating Z-Score rule scores...")
    rule_scores = calculate_z_score_rule(X_test)

    # Print some stats to verify scores aren't all zero
    print(f"\nScore Distributions (Mean | Max):")
    print(f"XGB Score:         {np.mean(xgb_scores):.4f} | {np.max(xgb_scores):.4f}")
    print(f"IsoForest Score:   {np.mean(iso_scores):.4f} | {np.max(iso_scores):.4f}")
    print(f"Z-Score Rule:      {np.mean(rule_scores):.4f} | {np.max(rule_scores):.4f}")
    print(f"Fraud cases in test set: {np.sum(y_test)}")

    # Evaluate individual layers
    print("\nIndividual Layer Performance (at 0.5 threshold):")
    for name, scores in [("XGB", xgb_scores), ("IsoForest", iso_scores), ("Z-Score Rule", rule_scores)]:
        pred = (scores > 0.5).astype(int)
        rec = recall_score(y_test, pred, zero_division=0)
        prec = precision_score(y_test, pred, zero_division=0)
        print(f"   {name:12}: Recall = {rec:.4f}, Precision = {prec:.4f}")

    # 4. Search for best weights
    print("\n5. Searching for best weights (finer grid, optimizing for F2 score)...")
    best_f2 = 0
    best_weights = None
    best_metrics = {}

    from sklearn.metrics import fbeta_score

    # Define weight ranges to iterate through (finer grid)
    for w1 in np.arange(0.0, 1.01, 0.02):
        for w2 in np.arange(0.0, 1.01 - w1, 0.02):
            w3 = 1.0 - (w1 + w2)
            w3 = round(w3, 2)
            if w3 < 0: continue

            final_score = (w1 * xgb_scores) + (w2 * iso_scores) + (w3 * rule_scores)
            y_pred = (final_score > 0.5).astype(int)
            
            curr_recall = recall_score(y_test, y_pred, zero_division=0)
            curr_precision = precision_score(y_test, y_pred, zero_division=0)
            curr_f1 = f1_score(y_test, y_pred, zero_division=0)
            # F2 score weights recall twice as much as precision
            curr_f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)

            # Optimize for F2 score primarily
            if curr_f2 > best_f2:
                best_f2 = curr_f2
                best_weights = {"w1_xgb": round(w1, 2), "w2_iso": round(w2, 2), "w3_rule": round(w3, 2)}
                best_metrics = {
                    "recall": round(curr_recall, 4),
                    "precision": round(curr_precision, 4),
                    "f1_score": round(curr_f1, 4),
                    "f2_score": round(curr_f2, 4)
                }

    print("\nBest Weight Combination Found (Optimized for F2):")
    print(f"Weights: {best_weights}")
    print(f"Metrics: {best_metrics}")

    # 5. NEW: Generate Business Impact Report
    print("\n7. Generating Business Impact Report...")
    
    # Use best weights to get final predictions for the report
    final_score_best = (best_weights["w1_xgb"] * xgb_scores) + \
                      (best_weights["w2_iso"] * iso_scores) + \
                      (best_weights["w3_rule"] * rule_scores)
    y_pred_best = (final_score_best > 0.5).astype(int)
    
    # Align amounts with predictions
    test_amounts = X_test['amount'].values
    
    # Financial Calculations
    total_fraud_amt = np.sum(test_amounts[y_test == 1])
    caught_fraud_amt = np.sum(test_amounts[(y_test == 1) & (y_pred_best == 1)])
    missed_fraud_amt = total_fraud_amt - caught_fraud_amt
    
    flagged_legit_amt = np.sum(test_amounts[(y_test == 0) & (y_pred_best == 1)])
    
    fp_count = np.sum((y_test == 0) & (y_pred_best == 1))
    tp_count = np.sum((y_test == 1) & (y_pred_best == 1))
    
    # Ratio of false blocks to 1 fraud catch
    fp_ratio = fp_count / tp_count if tp_count > 0 else float('inf')
    
    business_metrics = {
        "total_fraud_value_at_risk": round(float(total_fraud_amt), 2),
        "total_fraud_value_blocked": round(float(caught_fraud_amt), 2),
        "total_fraud_value_missed": round(float(missed_fraud_amt), 2),
        "percentage_value_saved": round(float(caught_fraud_amt / total_fraud_amt * 100), 2) if total_fraud_amt > 0 else 0,
        "customer_friction_value": round(float(flagged_legit_amt), 2),
        "false_positive_count": int(fp_count),
        "true_positive_count": int(tp_count),
        "block_to_catch_ratio": round(float(fp_ratio), 2)
    }

    print("-" * 40)
    print("BUSINESS IMPACT SUMMARY")
    print("-" * 40)
    print(f"Total Fraud Value in Test Set : ${business_metrics['total_fraud_value_at_risk']:,.2f}")
    print(f"Total Value Successfully Saved: ${business_metrics['total_fraud_value_blocked']:,.2f} ({business_metrics['percentage_value_saved']}%)")
    print(f"Total Value Missed (Loss)     : ${business_metrics['total_fraud_value_missed']:,.2f}")
    print(f"Value of Legit Funds Flagged  : ${business_metrics['customer_friction_value']:,.2f}")
    print(f"Block-to-Catch Ratio          : {business_metrics['block_to_catch_ratio']} legit blocks per 1 fraud catch")
    print("-" * 40)

    # 6. Save best weights and business metrics
    print("\n8. Saving results to file...")
    save_data = {
        "best_weights": best_weights,
        "ml_metrics": best_metrics,
        "business_impact": business_metrics,
        "search_parameters": {
            "test_size": 0.20,
            "random_state": 42,
            "threshold": 0.5,
            "primary_metric": "f2_score"
        }
    }
    
    save_path = os.path.join(PROJECT_ROOT, "models", "best_weights.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=4)
    
    print(f"Weights saved to: {save_path}")
    print("============ Finished ============")