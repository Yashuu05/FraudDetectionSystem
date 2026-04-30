from sklearn.model_selection import GridSearchCV
import json
import os
from src.pipelines.xgb_pipeline import build_xgb_pipeline
from src.pipelines.IsoForest_pipeline import build_IsoForest_pipeline

def build_full_pipeline(cat_cols, num_cols, model_name="xgb"):
    """
    Builds the full pipeline with GridSearchCV for hyperparameter tuning.
    Args:
        cat_cols: Categorical columns
        num_cols: Numerical columns
        model_name: "xgb" or "isoforest"
    Returns:
        GridSearchCV object for the selected model
    """
    try: 
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)

        if model_name.lower() == "xgb":
            # get full XGB pipeline
            xgb_pipeline = build_xgb_pipeline(cat_cols, num_cols)
            
            # Construct the path to the JSON file
            file_path = os.path.join(project_root, 'XGB_hyperparameters.json')

            # Open and load the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                xgb_params = json.load(file)

            # build GridSearchCV for hyperparameter tuning
            grid = GridSearchCV(
                estimator=xgb_pipeline,
                param_grid=xgb_params,
                cv=3,
                n_jobs=-1,
                verbose=2, 
                scoring="recall",
            )
            return grid

        elif model_name.lower() in ["isoforest", "isolation_forest"]:
            # get isoForest pipeline
            iso_pipeline = build_IsoForest_pipeline(categorical_cols=cat_cols, numerical_cols=num_cols)
            
            # import hyperparameters
            forest_file_path = os.path.join(project_root, 'IsoForest_hyperparameters.json')
            
            with open(forest_file_path, 'r', encoding='utf-8') as file:
                iso_params = json.load(file)
            
            # create GridSearchCV
            grid = GridSearchCV(
                estimator=iso_pipeline,
                param_grid=iso_params,
                scoring="recall",
                cv=3,
                verbose=2,
                n_jobs=-1
            )
            return grid
        
        else:
            raise ValueError(f"Model {model_name} not supported.")

    except Exception as e:
        print(f"Error in build_full_pipeline: {str(e)}")
        return None