"""
Week 3 Task: Unsupervised Learning & Model Evaluation
Author: Preksha B Angadi
AI Pioneers Internship (Machine Learning Internship)

This script demonstrates:
1. K-Means Clustering
2. Hierarchical (Agglomerative) Clustering
3. Dimensionality Reduction using PCA
4. Cross-validation, Confusion Matrix, Precision/Recall/F1/ROC-AUC
5. Hyperparameter tuning using GridSearchCV
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_classification
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, confusion_matrix, precision_score,
                              recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# PART A: CLUSTERING
# ---------------------------------------------------------
print("=" * 60)
print("PART A: CLUSTERING")
print("=" * 60)

X, y_true = make_blobs(n_samples=300, centers=4, n_features=5,
                        cluster_std=1.2, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
print(f"\nK-Means Silhouette Score: {kmeans_sil:.3f}")

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=4)
hier_labels = hierarchical.fit_predict(X_scaled)
hier_sil = silhouette_score(X_scaled, hier_labels)
print(f"Hierarchical Clustering Silhouette Score: {hier_sil:.3f}")

# Dimensionality Reduction with PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis")
axes[0].set_title(f"K-Means Clusters (Silhouette={kmeans_sil:.2f})")
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, cmap="plasma")
axes[1].set_title(f"Hierarchical Clusters (Silhouette={hier_sil:.2f})")
plt.tight_layout()
plt.savefig("clustering_results.png", dpi=120)
print("\nCluster visualization saved as clustering_results.png")

# ---------------------------------------------------------
# PART B: MODEL EVALUATION
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("PART B: MODEL EVALUATION")
print("=" * 60)

Xc, yc = make_classification(n_samples=400, n_features=8, n_informative=5,
                              random_state=42)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.25, random_state=42
)

# Cross-validation
base_model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(base_model, Xc_train, yc_train, cv=5)
print(f"\n5-Fold Cross-Validation Scores: {np.round(cv_scores, 3)}")
print(f"Mean CV Accuracy: {cv_scores.mean():.3f}")

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                            param_grid, cv=3, scoring="accuracy")
grid_search.fit(Xc_train, yc_train)
print(f"\nBest Parameters found by GridSearchCV: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(Xc_test)
y_proba = best_model.predict_proba(Xc_test)[:, 1]

cm = confusion_matrix(yc_test, y_pred)
precision = precision_score(yc_test, y_pred)
recall = recall_score(yc_test, y_pred)
f1 = f1_score(yc_test, y_pred)
roc_auc = roc_auc_score(yc_test, y_proba)

print(f"\n--- Final Tuned Model Performance on Test Set ---")
print(f"Confusion Matrix:\n{cm}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-Score : {f1:.3f}")
print(f"ROC-AUC  : {roc_auc:.3f}")

fpr, tpr, _ = roc_curve(yc_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned Random Forest")
plt.legend()
plt.savefig("roc_curve.png", dpi=120)
print("\nROC curve saved as roc_curve.png")

print("\nWeek 3 task completed successfully.")
