"""
Week 2 Task: Supervised Machine Learning Models
Author: Preksha B Angadi
AI Pioneers Internship (Machine Learning Internship)

This script demonstrates:
1. Logistic Regression (classification)
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors Classifier
5. Linear Regression (regression)
6. Model evaluation and comparison
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, mean_squared_error, r2_score)
from sklearn.datasets import make_classification, make_regression

# ---------------------------------------------------------
# PART A: CLASSIFICATION TASK
# ---------------------------------------------------------
print("=" * 60)
print("PART A: CLASSIFICATION - Predicting Customer Purchase")
print("=" * 60)

X_cls, y_cls = make_classification(
    n_samples=300, n_features=6, n_informative=4,
    n_redundant=1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Model": name, "Accuracy": round(acc, 3), "Precision": round(prec, 3),
        "Recall": round(rec, 3), "F1-Score": round(f1, 3)
    })

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-Score : {f1:.3f}")
    print(f"Confusion Matrix:\n{cm}")

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\n--- Model Comparison Table (Classification) ---")
print(results_df.to_string(index=False))
results_df.to_csv("classification_model_comparison.csv", index=False)

# ---------------------------------------------------------
# PART B: REGRESSION TASK
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("PART B: REGRESSION - Predicting a Continuous Target")
print("=" * 60)

X_reg, y_reg = make_regression(n_samples=300, n_features=4, noise=15, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(Xr_train, yr_train)
yr_pred = lin_reg.predict(Xr_test)

mse = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)
r2 = r2_score(yr_test, yr_pred)

print(f"\nLinear Regression Results:")
print(f"MSE : {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")

best_model = results_df.iloc[0]["Model"]
print(f"\nBest performing classification model: {best_model}")
print("\nWeek 2 task completed successfully.")
