from xgboost import XGBClassifier
from src.piplelines.xgb_pipeline import build_xgb_model
from src.piplelines.model_pipeline import build_pipeline
from sklearn.pipeline import Pipeline


def build_xgb_pipeline(cat_cols, num_cols):
    try:
        model = XGBClassifier(
            random_state = 42
        )
        # Get the pipeline
        pipeline = build_pipeline(cat_cols, num_cols)
        # build XGBClassifier pipeline
        xgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", pipeline),
                ("model", model)
            ])

        return xgb_pipeline

    except Exception as e:
        print(f"{str(e)}")
        return None