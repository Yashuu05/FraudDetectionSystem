import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from app.system_1.script import generate_data
import time

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("system1.html")

@app.route("/generate_transaction", methods=["POST"])
def generate_transaction():
    try:
        data_input, data = generate_data()
        # In a real app, we would pass data_input to the model (system_2) here.
        # For now, we just return the display data.
        return jsonify(data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
