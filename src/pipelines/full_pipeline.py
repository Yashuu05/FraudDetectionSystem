from sklearn.model_selection import GridSearchCV
import json
import os
from src.pipelines.xgb_pipeline import build_xgb_pipeline
from src.pipelines.IsoForest_pipeline import build_IsoForest_pipeline

def build_full_pipeline(cat_cols, num_cols):

    try: 
        # get full XGB pipeline
        xgb_pipeline = build_xgb_pipeline(cat_cols, num_cols)
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the JSON file
        file_path = os.path.join(os.path.dirname(script_dir), 'XGB_hyperparameters.json')

        # Open and load the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        xgb_params = data

        # build GridSearchCV for hyperparameter tuning
        xgb_grid = GridSearchCV(
            estimator=xgb_pipeline,
            param_grid=xgb_params,
            cv=3,
            n_jobs=-1,
            verbose=2, 
            scoring="recall",
        )
        
        # Isolation forest full pipeline
    
        # get isoForest pipeline
        isoForest_pipeline = build_IsoForest_pipeline(categorical_cols=cat_cols, numerical_cols=num_cols)
        # import hyperparameters
        forest_file_path = os.path.join(os.path.dirname(script_dir), 'IsoForest_hyperparameters.json')
        with open(forest_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        isoForest_params = data
        
        # create GridSearchCV
        isoForest_grid = GridSearchCV(
            estimator=isoForest_pipeline,
            param_grid=isoForest_params,
            scoring="recall",
            cv=3,
            verbose=2,
            n_jobs=-1
        )
        
        return xgb_grid, isoForest_grid

    except Exception as e:
        print(f"Error! {str(e)}")
        return None