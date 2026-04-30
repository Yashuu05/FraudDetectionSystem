"""
Purpose: to use mean and stdev for z-score calculation
"""
import os
import sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
sys.path.insert(0, root)
PROJECT_ROOT = root 
from src.utils.data_utils import load_dataset_sample
import statistics

def calculate_mean_std(df):
    amount = df['amount']
    mean = statistics.mean(amount)
    std = statistics.stdev(amount)
    return mean, std    

if __name__ == "__main__":
    # create an empty dictionary to store constants
    metrics = {}

    # load dataset
    print("loading dataset...")
    data_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")
    dataset = load_dataset_sample(
        file_path=data_path,
        sample_size=100000,
        random_state=42,
        chunksize=50000
    )

    # caclulate mean and standard deviation
    print("\ncalculating mean and stdev of dataset...")
    mean, std = calculate_mean_std(df=dataset)
    # store in the dictionary
    metrics["mean"] = mean
    metrics["std"] = std
    file_save_path = os.path.join(PROJECT_ROOT, "app", "system_2","constants.json")
    os.makedirs(os.path.dirname(file_save_path), exist_ok=True)

    print("saving constants...")
    with open(file=file_save_path, mode="+w") as f:
        json.dump(metrics, f)
    print(f"constants saved successfully in {file_save_path}")