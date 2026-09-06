# Customer Churn Prediction & Business Intelligence System

An end-to-end group project that takes a raw telecom customer churn dataset all the way to a deployed prediction prototype and an interactive BI dashboard — covering data engineering, exploratory analysis, feature engineering, machine learning, model explainability, retention strategy, a live prediction app, and business intelligence reporting.

## Overview

Customer churn — when a customer stops using a company's product or service — is far more expensive to recover from than it is to prevent, since acquiring a replacement customer costs more than retaining an existing one. This project analyzes a telecom customer churn dataset (7,043 customers, IBM Telco Customer Churn structure) to identify which customers are likely to churn, explain *why*, and turn those findings into concrete retention actions — packaged into both a working prediction interface and a business intelligence dashboard.

The project was built by a 26-member team split into seven technical groups plus a Final Report team, with each group building directly on the previous group's output.

## Pipeline

| Stage | Folder | What it does |
|---|---|---|
| 1. Data Engineering | `Dataset selection and understanding/`, `Data Preprocessing & Final Validation/`, `Cleaned Dataset/` | Dataset selection, data dictionary, cleaning, encoding, scaling, and a full data quality report |
| 2. EDA & Business Analysis | `EDA/`, `Figures/` | Churn distribution, demographic/tenure/contract/service analysis, statistical tests, business-question answers |
| 3. Feature Engineering & Statistics | `Feature Engineering & Statistical Analysis/` | Engineered features, Chi-square/t-test/Mutual Information testing, multicollinearity checks, final feature set |
| 4. Machine Learning | `Machine Learning/` | Logistic Regression, Decision Tree, Random Forest and XGBoost trained and compared; final tuned model selected and evaluated on a held-out test set |
| 5. Explainability & Retention Strategy | `Model Explainability & Retention Strategy/` | SHAP/odds-ratio explainability, customer risk tiers, a costed churn-driver → segment → action retention playbook |
| 6. Prediction Interface | `prediction-interface/` | Streamlit app for live, single-customer churn prediction, risk classification and recommended actions |
| 7. BI Dashboard | `Dashboard/` | Power BI dashboard (Executive Overview, Churn Analysis, Customer Risk Analysis pages) |

## Key Results

- **Final model:** Tuned Logistic Regression — held-out test F1-score **0.62**, ROC-AUC **0.85**, Recall **0.79**
- **Top churn drivers:** Contract type and tenure (month-to-month churn 42.7% vs. 2.8% for two-year contracts; new customers churn ~47% vs. ~9.5% for 37+ month customers)
- **Risk segmentation:** 33% of customers classified High Risk, with an estimated retention-campaign ROI of ~1,070%

## Tech Stack

Python (Pandas, NumPy, scikit-learn, XGBoost, SHAP), Streamlit, Power BI, Jupyter Notebooks.

