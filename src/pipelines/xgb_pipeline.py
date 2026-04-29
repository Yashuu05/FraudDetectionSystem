from xgboost import XGBClassifier
from src.pipelines.model_pipeline import build_pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


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
                ("pca", PCA(n_components=13))
                ("model", model)
            ])

        return xgb_pipeline

    except Exception as e:
        print(f"{str(e)}")
        return None