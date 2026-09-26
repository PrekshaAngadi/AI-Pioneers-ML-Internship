# AI Pioneers Internship – Machine Learning Internship

**Intern:** Preksha B Angadi
**Program:** AI Pioneers Internship (Machine Learning Internship) — 4 Week Program

This repository contains all weekly deliverables for the AI Pioneers Machine
Learning Internship: data preprocessing, supervised learning, unsupervised
learning & model evaluation, and a final deployed ML capstone project.

## Repository Structure

```
AI-Pioneers-ML-Internship/
│
├── Week1_Data_Preprocessing/
│   ├── preprocessing.py
│   ├── sample_dataset_raw.csv
│   ├── sample_dataset_preprocessed.csv
│   └── README.md
│
├── Week2_Supervised_Learning/
│   ├── supervised_models.py
│   ├── classification_model_comparison.csv
│   └── README.md
│
├── Week3_Unsupervised_Learning/
│   ├── clustering_evaluation.py
│   ├── clustering_results.png
│   ├── roc_curve.png
│   └── README.md
│
├── Week4_Deployment_Capstone/
│   ├── model_training.py
│   ├── app.py
│   ├── capstone_dataset.csv
│   ├── model.pkl
│   ├── scaler.pkl
│   └── README.md
│
├── requirements.txt
└── README.md   (this file)
```

## How to Run

```bash
pip install -r requirements.txt

# Week 1
cd Week1_Data_Preprocessing && python preprocessing.py

# Week 2
cd ../Week2_Supervised_Learning && python supervised_models.py

# Week 3
cd ../Week3_Unsupervised_Learning && python clustering_evaluation.py

# Week 4 — train then serve
cd ../Week4_Deployment_Capstone
python model_training.py
python app.py
# then in another terminal:
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"Age\": 35, \"EstimatedSalary\": 60000}"
```

## Weekly Summary

| Week | Topic | Key Deliverable |
|------|-------|------------------|
| 1 | Python for ML & Data Preprocessing | Cleaned & preprocessed sample dataset |
| 2 | Supervised Machine Learning Models | Trained & compared 4 classification models + 1 regression model |
| 3 | Unsupervised Learning & Model Evaluation | K-Means/Hierarchical clustering, cross-validation, ROC-AUC, GridSearchCV |
| 4 | AI Project Deployment & Capstone | End-to-end ML app deployed via Flask REST API |

## Author
**Preksha B Angadi**
