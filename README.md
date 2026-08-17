# Customer Churn Prediction & Business Intelligence System

A group project that builds an end-to-end pipeline for predicting customer churn and turning the results into actionable business intelligence — covering data engineering, exploratory analysis, feature engineering, machine learning, model explainability, a prediction prototype, and a BI dashboard.

## Overview

Customer churn — when a customer stops using a company's product or service — is costly to acquire replacements for. This project analyzes a customer churn dataset to identify which customers are likely to churn, explain *why*, and recommend retention actions, packaged into both a prediction interface and a business intelligence dashboard.

## Repository Structure

```
customer-churn-project/
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda_business_analysis.ipynb
│   ├── 03_feature_engineering_statistics.ipynb
│   ├── 04_machine_learning.ipynb
│   └── 05_explainability_retention.ipynb
├── models/
│   └── final_model.pkl
├── app/                  # Group 6 — prediction interface
├── dashboard/            # Group 7 — BI dashboard files/screenshots
├── report/               # Final report and supporting docs
├── requirements.txt
└── README.md
```
