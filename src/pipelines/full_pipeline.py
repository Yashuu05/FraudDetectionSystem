from sklearn.model_selection import GridSearchCV
import json
import os
from pipelines.xgb_pipeline import build_xgb_pipeline


def build_full_pipeline(cat_cols, num_cols):

    try: 
        # get full XGB pipeline
        xgb_pipeline = build_xgb_pipeline(cat_cols, num_cols)
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the JSON file
        file_path = os.path.join(script_dir, 'src', 'hyperparameters.json')

        # Open and load the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        params = data

        # build GridSearchCV for hyperparameter tuning
        xgb_grid = GridSearchCV(
            estimator=xgb_pipeline,
            param_grid=params,
            cv=3,
            n_jobs=-1,
            verbose=2, 
            scoring="recall",
        )
    
        return xgb_grid

    except Exception as e:
        print(f"Error! {str(e)}")
        return None