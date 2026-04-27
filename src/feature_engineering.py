from src.utils import data_utils
import pandas as pd
import os

def build_features(df):
    # Define high risk types (typically TRANSFER and CASH_OUT)
    df["high_risk_type"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

    df["sender_balance_error"] = abs(
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )

    # Receiver balance error
    df["receiver_balance_error"] = abs(
        df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    )

    # Any balance inconsistency
    df["balance_error_flag"] = (
        (df["sender_balance_error"] > 0) |
        (df["receiver_balance_error"] > 0)
    ).astype(int)

    # Missing balance info
    df["missing_sender_balance"] = (
        (df["oldbalanceOrg"] == 0) &
        (df["newbalanceOrig"] == 0)
    ).astype(int)

    df["missing_receiver_balance"] = (
        (df["oldbalanceDest"] == 0) &
        (df["newbalanceDest"] == 0)
    ).astype(int)

    # Destination balance unchanged
    df["dest_balance_unchanged"] = (
        (df["amount"] > 0) &
        (df["oldbalanceDest"] == df["newbalanceDest"])
    ).astype(int)

    # Sender balance unchanged
    df["sender_balance_unchanged"] = (
        (df["amount"] > 0) &
        (df["oldbalanceOrg"] == df["newbalanceOrig"])
    ).astype(int)

    # Large transaction flag
    df["large_transaction"] = (df["amount"] > 200000).astype(int)

    # Amount vs sender balance (handle division by zero)
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-9)

    # Amount vs receiver balance (handle division by zero)
    df["amount_to_dest_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1e-9)

    # Number of transactions per step
    df["transactions_per_step"] = df.groupby("step")["amount"].transform("count")

    # Average transaction amount per step
    df["avg_amount_per_step"] = df.groupby("step")["amount"].transform("mean")

    # Strong fraud signal
    df["high_risk_combination"] = (
        (df["balance_error_flag"] == 1) &
        (df["dest_balance_unchanged"] == 1) &
        (df["high_risk_type"] == 1)
    ).astype(int)

    # Another strong pattern
    df["zero_balance_transfer"] = (
        (df["oldbalanceOrg"] == 0) &
        (df["amount"] > 0)
    ).astype(int)

    return df


if __name__ == "__main__":
    # 1: load dataset
    raw_data_path = r"D:\projects\FraudDetectionSystem\Data\raw\Synthetic_Financial_datasets_log.csv"
    
    print(f"Loading dataset from: {raw_data_path}")
    df = data_utils.load_dataset(file_path=raw_data_path)

    if df is not None:
        # 2: feature engineering
        print("Performing feature engineering...")
        processed_df = build_features(df.copy())
        
        # 3. save data 
        save_file_path = "Data/processed/featured_data.csv"
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        
        print(f"Saving processed data to: {save_file_path}")
        data_utils.save_dataset(data=processed_df, file_path=save_file_path)

        print("\nNew dataset info:")
        processed_df.info()
    else:
        print("Failed to load dataset.")