"""
Week 1 Task: Python for Machine Learning & Data Preprocessing
Author: Preksha B Angadi
AI Pioneers Internship (Machine Learning Internship)

This script demonstrates:
1. Creating / loading a sample dataset
2. Handling missing values
3. Encoding categorical variables
4. Feature scaling / normalization
5. Exploratory Data Analysis (EDA)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ---------------------------------------------------------
# 1. Create a sample dataset (Customer purchase dataset)
# ---------------------------------------------------------
np.random.seed(42)

data = {
    "CustomerID": range(1, 21),
    "Age": [25, 32, np.nan, 45, 29, 38, np.nan, 51, 22, 34,
            41, np.nan, 27, 36, 48, 30, np.nan, 55, 23, 39],
    "Gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Female",
               "Male", "Female", "Male", "Female", "Male", "Female", "Male",
               "Female", "Male", "Female", "Male", "Female", "Male"],
    "Salary": [40000, 52000, 58000, np.nan, 45000, 62000, 39000, 71000,
               43000, 55000, 67000, 48000, np.nan, 51000, 73000, 46000,
               50000, 80000, 41000, 60000],
    "City": ["Bangalore", "Mumbai", "Delhi", "Bangalore", "Chennai", "Mumbai",
             "Delhi", "Bangalore", "Chennai", "Mumbai", "Delhi", "Bangalore",
             "Chennai", "Mumbai", "Delhi", "Bangalore", "Chennai", "Mumbai",
             "Delhi", "Bangalore"],
    "Purchased": [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
df.to_csv("sample_dataset_raw.csv", index=False)
print("Raw dataset saved as sample_dataset_raw.csv")
print("\n--- Raw Data Preview ---")
print(df.head())

# ---------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values Per Column ---")
print(df.isnull().sum())

print("\n--- Summary Statistics ---")
print(df.describe())

# ---------------------------------------------------------
# 3. Handling Missing Values
# ---------------------------------------------------------
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

print("\n--- Missing Values After Imputation ---")
print(df.isnull().sum())

# ---------------------------------------------------------
# 4. Encoding Categorical Variables
# ---------------------------------------------------------
le_gender = LabelEncoder()
df["Gender_Encoded"] = le_gender.fit_transform(df["Gender"])

df = pd.get_dummies(df, columns=["City"], prefix="City")

print("\n--- Data After Encoding ---")
print(df.head())

# ---------------------------------------------------------
# 5. Feature Scaling / Normalization
# ---------------------------------------------------------
scaler = MinMaxScaler()
df[["Age_Scaled", "Salary_Scaled"]] = scaler.fit_transform(df[["Age", "Salary"]])

std_scaler = StandardScaler()
df[["Age_Standardized", "Salary_Standardized"]] = std_scaler.fit_transform(df[["Age", "Salary"]])

print("\n--- Data After Scaling ---")
print(df[["Age", "Age_Scaled", "Age_Standardized",
          "Salary", "Salary_Scaled", "Salary_Standardized"]].head())

# ---------------------------------------------------------
# 6. Save the cleaned/preprocessed dataset
# ---------------------------------------------------------
df.to_csv("sample_dataset_preprocessed.csv", index=False)
print("\nPreprocessed dataset saved as sample_dataset_preprocessed.csv")
print("\nWeek 1 task completed successfully.")
