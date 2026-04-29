import os
import pandas as pd
import sys

# Get the directory where test.py is located
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# load data
data_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")

if os.path.exists(data_path):
    print(f"Loading dataset from: {data_path}")
    # Using nrows=100000 to avoid loading the entire 1.1GB file if only testing
    df = pd.read_csv(data_path, nrows=100000)
    print("\nDataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nColumn Info:")
    print(df.info())
else:
    print(f"Error: Dataset not found at {data_path}")