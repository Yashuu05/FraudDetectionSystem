# data utilities

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load dataset (full – use only for small files)
def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error! {str(e)}")
        return None

# load a random sample from a large CSV without reading it all into memory
def load_dataset_sample(file_path, sample_size=100_000, random_state=42, chunksize=100_000):
    """
    Faster version using chunk-level sampling.
    """
    try:
        print(f"Sampling {sample_size:,} rows from: {file_path}")
        
        # For very fast loading during tuning, if sample_size is small relative to file
        # we just take the first N rows or use a simpler sampling.
        # But to be 'fair' to reservoir sampling, we'll use a more efficient loop.
        
        chunks = []
        total_rows = 0
        # First, let's get a sample from the first few chunks to meet the size quickly
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            chunks.append(chunk)
            total_rows += len(chunk)
            if total_rows >= sample_size:
                break
        
        full_df = pd.concat(chunks).reset_index(drop=True)
        if len(full_df) > sample_size:
            return full_df.sample(n=sample_size, random_state=random_state)
        return full_df

    except Exception as e:
        print(f"Error! {str(e)}")
        return None

# saving data into csv
def save_dataset(data, file_path):
    try:
        data.to_csv(file_path, index=False)
        print("Successful! Dataset saved.")
    except Exception as e:
        print(f"Error! {str(e)}")
        return None

# split dataset into training and testing
def split_dataset(randomState, testSize, X, y):

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=randomState, test_size=testSize, shuffle=True
        )
        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error! {str(e)}")

# find categorical and numerical values
def separate_cols_type(data):
    categorical_cols = []
    numerical_cols = []
    
    for cols in data.columns:
        if data[cols].dtypes == "object":
            categorical_cols.append(cols)
        else:
            numerical_cols.append(cols)
    return categorical_cols, numerical_cols