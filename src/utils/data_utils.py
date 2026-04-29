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
def load_dataset_sample(file_path, sample_size=100_000, random_state=42, chunksize=50_000):
    """
    Streams the CSV in chunks of `chunksize` rows and draws a random sample
    of up to `sample_size` rows.  Memory usage is bounded by
    O(chunksize + sample_size) regardless of total file size.
    """
    try:
        rng = np.random.default_rng(random_state)
        reservoir = []          # collected sample rows
        total_seen = 0          # rows processed so far

        print(f"Sampling {sample_size:,} rows from: {file_path}")

        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            chunk_len = len(chunk)

            for i, row in chunk.iterrows():
                total_seen += 1
                if len(reservoir) < sample_size:
                    reservoir.append(row)
                else:
                    # reservoir sampling: replace a random earlier row
                    j = int(rng.integers(0, total_seen))
                    if j < sample_size:
                        reservoir[j] = row

        sample_df = pd.DataFrame(reservoir).reset_index(drop=True)
        print(f"Sample ready: {len(sample_df):,} rows from {total_seen:,} total.")
        return sample_df

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