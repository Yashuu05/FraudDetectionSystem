import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, ClassifierMixin
from src.pipelines.model_pipeline import build_pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

class IsolationForestWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for IsolationForest to make it compatible with supervised metrics.
    Maps Isolation Forest output (-1 for outlier, 1 for inlier) 
    to standard fraud labels (1 for fraud/outlier, 0 for legit/inlier).
    """
    def __init__(self, n_estimators=100, contamination=0.1, max_samples='auto', random_state=42, n_jobs=-1, verbose=0):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.model = None

    def fit(self, X, y=None):
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        self.model.fit(X)
        return self

    def predict(self, X):
        # Isolation Forest returns -1 for outliers and 1 for inliers
        # Standard fraud labels: 1 for fraud (outliers), 0 for legit (inliers)
        raw_preds = self.model.predict(X)
        return np.where(raw_preds == -1, 1, 0)

    def decision_function(self, X):
        return self.model.decision_function(X)

def build_IsoForest_pipeline(categorical_cols, numerical_cols):
    # get the raw pipeline
    preprocessor = build_pipeline(cat_cols=categorical_cols, num_cols=numerical_cols)
    
    # define model wrapper
    model_wrapper = IsolationForestWrapper(
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    
    # build pipeline
    iso_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=13)),
        ('model', model_wrapper)
    ])

    return iso_pipeline