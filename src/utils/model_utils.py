import joblib as jb

#load model
def load_model(file_path):
    try:
        return jb.load(file_path)
    except Exception as e:
        print(f"Error! {str(e)}")
        return None
    
# save model
def save_model(file_path, model):
    try:
        jb.dump(value=model, filename=file_path)
        return f"Successul! Model saved."
    except Exception as e:
        print(f"Error! {str(e)}")
        return None
    
# evaluating model's perfromance for supervised model
def evalulate_model(y_test, y_pred):
    """
    this function evaluates' performance of trained supervised model
    Input: y_test, y_pred 
    Output: list of recall, precision, f1 socre, accuracy, roc_auc_score and classification report
    """
    
    from sklearn.metrics import  accuracy_score, precision_score, roc_auc_score, recall_score, f1_score, classification_report
    
    try:
        # 1
        recall = recall_score(y_test, y_pred, average="weighted")
        # 2
        precision = precision_score(y_test, y_pred, average="weighted")
        # 3
        f1 = f1_score(y_test, y_pred, average="weighted")
        # 4
        acc = accuracy_score(y_test, y_pred)
        # 5
        roc = roc_auc_score(y_test, y_pred, average="weighted")
        # 6
        report = classification_report(y_test, y_pred)
        results = [recall, precision, f1, acc, roc, report]
        
        return results
    
    except Exception as e:
        print(f"Error! {str(e)}")