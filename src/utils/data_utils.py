# data utilities

import pandas as pd
from sklearn.model_selection import train_test_split
# load dataset
def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=randomState, test_size=testSize, shuffle=True)
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error! {str(e)}")