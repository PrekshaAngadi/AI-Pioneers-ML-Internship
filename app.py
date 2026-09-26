"""
Week 4 Task: AI Project Deployment & Capstone
Author: Preksha B Angadi
AI Pioneers Internship (Machine Learning Internship)

A simple Flask API that serves the trained RandomForest model
(model.pkl / scaler.pkl produced by model_training.py) and returns
a purchase prediction for a given Age and EstimatedSalary.

Run:
    python model_training.py   # trains and saves model.pkl, scaler.pkl
    python app.py               # starts the API on http://127.0.0.1:5000

Test:
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"Age": 35, "EstimatedSalary": 60000}'
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Pioneers Internship - Week 4 Capstone API",
        "usage": "POST /predict with JSON body {'Age': <int>, 'EstimatedSalary': <float>}"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        age = float(data["Age"])
        salary = float(data["EstimatedSalary"])

        features = np.array([[age, salary]])
        features_scaled = scaler.transform(features)

        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])

        return jsonify({
            "Age": age,
            "EstimatedSalary": salary,
            "Purchased_Prediction": prediction,
            "Purchase_Probability": round(probability, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
