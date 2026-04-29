from src.utils import data_utils
from src.utils import model_utils
from src.pipelines import full_pipeline
import os 
import mlflow

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mlflow.set_experiment("fruad detection")
mlflow.autolog()

# step 1 load dataset
print("1. loading dataset...")
data_path = os.path.join(PROJECT_ROOT, "Data", "processed", "featured_data.csv")
dataset = data_utils.load_dataset_sample(
    file_path=data_path,
    sample_size=100000,
    random_state=42,
    chunksize=50000
)

if dataset is None:
    raise FileNotFoundError(f"Error! dataset {data_path} couldn't load.")
    
else:

    try:
        df = dataset.copy()
        print("2. Splitting dataset...")

        X = df.drop(["isFraud", "isFlaggedFraud","nameOrig","nameDest"])
        y = df['isFraud']
        X_train, X_test, y_train, y_test = data_utils.split_dataset(
            randomState=42,
            testSize=0.20,
            X=X,
            y=y
        )

        print("3. loading model pipeline...")
        categorical_cols, numerical_cols = data_utils.separate_cols_type(data=df)
        model_pipeline = full_pipeline.build_full_pipeline(cat_cols=categorical_cols, num_cols=numerical_cols)

        print("4. fitting model")
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        print ("5. Evaluating model...")
        metrics = ["recall", "precision", "f1", "acc", "roc", "report"]
        results = model_utils.evalulate_model(y_test=y_test, y_pred=y_pred)
        for i,j in zip(metrics, results):
                print(f"{i} : {j} \n")

    except Exception as e:
            print(f"Error! {str(e)}")