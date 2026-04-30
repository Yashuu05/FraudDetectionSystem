from sklearn.ensemble import IsolationForest
from src.pipelines.model_pipeline import build_pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def build_IsoForest_pipeline(categorical_cols, numerical_cols):

    # get the raw pipeline
    preprcoessor = build_pipeline(cat_cols=categorical_cols, num_cols=numerical_cols)
    # define model
    isolation_forest = IsolationForest(
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    # buid pipleline
    IsoForest_pipeline = Pipeline(steps=[
        ('preprocessor', preprcoessor),
        ('pca', PCA(n_components=13)),
        ('model', isolation_forest)
    ])

    return IsoForest_pipeline