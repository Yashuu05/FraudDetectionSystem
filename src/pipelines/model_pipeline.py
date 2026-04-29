from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler


def build_pipeline(cat_cols, num_cols):
    
    cat_preprocessor = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_preprocessor = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler())
    ])

    pipeline = ColumnTransformer(transformers=[
        ('cat', cat_preprocessor, cat_cols),
        ('num', num_preprocessor, num_cols)
    ])

    return pipeline