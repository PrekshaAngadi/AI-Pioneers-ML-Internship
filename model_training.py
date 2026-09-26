"""
Week 4 Task: AI Project Deployment & Capstone
Author: Preksha B Angadi
AI Pioneers Internship (Machine Learning Internship)

Trains a final model on a synthetic "customer purchase prediction" dataset
and serializes it with joblib so it can be served by the Flask API (app.py).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# ---------------------------------------------------------
# 1. Create dataset (Age, EstimatedSalary -> Purchased)
# ---------------------------------------------------------
np.random.seed(42)
X, y = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)

# Rescale features to look like realistic Age / Salary values
age = (X[:, 0] * 10 + 40).clip(18, 65)
salary = (X[:, 1] * 15000 + 55000).clip(15000, 150000)

df = pd.DataFrame({"Age": age, "EstimatedSalary": salary, "Purchased": y})
df.to_csv("capstone_dataset.csv", index=False)
print("Dataset saved as capstone_dataset.csv")

# ---------------------------------------------------------
# 2. Train / test split + scaling
# ---------------------------------------------------------
X = df[["Age", "EstimatedSalary"]].values
y = df["Purchased"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. Train model
# ---------------------------------------------------------
model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 4. Serialize model + scaler for the Flask API
# ---------------------------------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nSaved model.pkl and scaler.pkl for deployment.")
print("\nWeek 4 model training completed successfully.")
