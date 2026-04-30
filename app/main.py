import os
import sys
import mlflow
from dotenv import load_dotenv

load_dotenv()
# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import redis
from flask import Flask, render_template, request, jsonify
from app.system_1.script import generate_data
from src.system2.risk_score_cal import load_all_models, load_constants, calculate_risk_score

app = Flask(__name__)
cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
)

# Initialize MLflow
mlflow.set_experiment("Fraud_Live_Monitoring")

# Initialize models and constants globally for the app
print("--- Initializing Fraud Detection Systems ---")
xgb_model, iso_model = load_all_models()
mean, std, w1, w2, w3 = load_constants()
print("--- System Ready ---")

@app.route("/")
def index():
    return render_template("system1.html")

@app.route("/generate_transaction", methods=["POST"])
def generate_transaction():
    try:
        # 1. Generate synthetic transaction data (including ground truth for tracking)
        data_input, display_data, is_fraud_actual = generate_data()
        
        # 2. Calculate Risk Score using System 2 (Multi-layered approach)
        risk_score, details = calculate_risk_score(
            data_input=data_input,
            mean=mean,
            stdev=std,
            w1=w1, w2=w2, w3=w3,
            xgb_model=xgb_model,
            iso_model=iso_model
        )
        
        # 3. Determine risk level and status
        if risk_score > 0.6:
            risk_level = "High"
            risk_status = "CRITICAL"
        elif risk_score > 0.4:
            risk_level = "Medium"
            risk_status = "WARNING"
        else:
            risk_level = "Low"
            risk_status = "SAFE"
            
        # 4. Log to MLflow for performance analysis
        # We use a single background run or log each prediction as an event
        with mlflow.start_run(run_name="Live_Transaction_Inference", nested=True):
            mlflow.log_metric("risk_score", risk_score)
            mlflow.log_metric("is_fraud_actual", 1 if is_fraud_actual else 0)
            mlflow.log_param("transaction_type", display_data["type"])
            mlflow.log_param("predicted_status", risk_status)
            
        # 5. Add risk data to display payload
        display_data["risk_score"] = risk_score
        display_data["risk_level"] = risk_level
        display_data["risk_status"] = risk_status
        display_data["score_details"] = details
        
        return jsonify(display_data)
        
    except Exception as e:
        print(f"Error in transaction generation: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    # Use environment variables or defaults
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
