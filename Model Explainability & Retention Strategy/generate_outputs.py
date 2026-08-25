"""
generate_outputs.py
===================
Master execution script for Group 5.
Generates all outputs, figures, metrics, and individual predictions.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR / "src"))

from explainers import ChurnModelExplainer
from risk_profiler import CustomerRiskProfiler
from retention_engine import RetentionStrategyEngine

# Setup directories
OUTPUTS_DIR = CURRENT_DIR / "outputs"
FIGURES_DIR = CURRENT_DIR / "figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Locate data
PROJECT_ROOT = CURRENT_DIR.parent
DATA_PATH = PROJECT_ROOT / "Feature Engineering & Statistical Analysis" / "outputs" / "customer_churn_feature_engineered.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "Cleaned Dataset" / "customer_churn_cleaned.xlsx"

print("=" * 70)
print("GROUP 5: MODEL EXPLAINABILITY & RETENTION STRATEGY PIPELINE")
print("=" * 70)

# 1. Load Data
print("\n[1/6] Loading feature engineered and cleaned dataset...")
df_feat = pd.read_csv(DATA_PATH)
df_clean = pd.read_excel(CLEANED_DATA_PATH)
print(f"Loaded {len(df_feat):,} records with {df_feat.shape[1]} features.")

CORE_FEATURES = [
    "Senior Citizen", "Partner", "Dependents", "Tenure Months", "Multiple Lines",
    "Internet Service", "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies", "Contract",
    "Paperless Billing", "Payment Method", "Monthly Charges", "Total Charges",
    "Fiber_Monthly_Risk", "New_High_Spend", "New_Monthly_Customer", "Security_Tech_Bundle"
]
TARGET = "Churn Value"

X = df_feat[CORE_FEATURES].copy()
y = df_feat[TARGET].copy()

# 2. Reconstruct / Fit Final Logistic Pipeline
print("\n[2/6] Initializing Final Churn Deployment Pipeline...")
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

clf = LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
pipeline.fit(X, y)

print("Pipeline fitted successfully on complete dataset.")

# 3. Global Explainability & SHAP Analysis (Member 1)
print("\n[3/6] Computing Global Feature Importance and SHAP Values (Member 1)...")
explainer = ChurnModelExplainer(pipeline, feature_names=CORE_FEATURES, background_data=X)

# Global coefficients and odds ratios
df_global_imp = explainer.compute_global_feature_importance()
df_global_imp.to_csv(OUTPUTS_DIR / "global_feature_importance.csv", index=False)
print("Saved global_feature_importance.csv")

# Global SHAP summary
df_shap_summary = explainer.compute_global_shap_summary(X)
df_shap_summary.to_csv(OUTPUTS_DIR / "global_shap_summary.csv", index=False)
print("Saved global_shap_summary.csv")

# Plots
explainer.plot_feature_importance(top_n=15, output_path=FIGURES_DIR / "global_feature_importance_odds_ratios.png")
explainer.plot_shap_summary_bar(X, top_n=15, output_path=FIGURES_DIR / "shap_global_bar.png")
explainer.plot_shap_beeswarm(X, top_n=15, output_path=FIGURES_DIR / "shap_summary_beeswarm.png")
print("Saved global explainability figures.")

# 4. Local Explanations & Risk Categorization (Member 2)
print("\n[4/6] Scoring Customers, Assigning Risk Tiers, and Decomposing Drivers (Member 2)...")
profiler = CustomerRiskProfiler(explainer)

# Score entire dataset
scored_df = profiler.score_and_profile_dataset(df_feat, CORE_FEATURES)

# Compute Risk Summary
risk_summary = profiler.compute_risk_summary(scored_df, df_original=df_clean)
risk_summary.to_csv(OUTPUTS_DIR / "customer_risk_summary.csv", index=False)
print("Saved customer_risk_summary.csv")

# High-Risk factor profiling
high_risk_factors = profiler.analyze_high_risk_factors(scored_df)
high_risk_factors.to_csv(OUTPUTS_DIR / "high_risk_profile_analysis.csv", index=False)
print("Saved high_risk_profile_analysis.csv")

# Risk plots
profiler.plot_risk_distribution(risk_summary, output_path=FIGURES_DIR / "risk_distribution_donut.png")
profiler.plot_high_risk_drivers(high_risk_factors, top_n=10, output_path=FIGURES_DIR / "high_risk_drivers_breakdown.png")
print("Saved risk profiling figures.")

# Individual waterfall plots for exemplar customers (e.g. C0001, C0002, C0003)
for idx in [0, 1, 2]:
    cid = scored_df.iloc[idx]["Customer_ID"]
    explainer.plot_waterfall(X.iloc[idx], customer_id=cid, max_display=8, output_path=FIGURES_DIR / f"individual_waterfall_{cid.lower()}.png")
print("Saved exemplar individual waterfall plots.")

# 5. Retention Strategy Matrix & Business Plays (Member 3)
print("\n[5/6] Generating Retention Strategy Matrix and Actionable Recommendations (Member 3)...")
retention_engine = RetentionStrategyEngine()

strategy_matrix = retention_engine.get_strategy_matrix()
strategy_matrix.to_csv(OUTPUTS_DIR / "churn_driver_action_matrix.csv", index=False)
print("Saved churn_driver_action_matrix.csv")

# Enhance customer-level table with actions
customer_explanations_full = retention_engine.enhance_customer_explanations(scored_df)
customer_explanations_full.to_csv(OUTPUTS_DIR / "customer_level_explanations.csv", index=False)
print("Saved customer_level_explanations.csv (7,043 scored customers)")

# Campaign ROI Simulation
roi_df = retention_engine.simulate_retention_campaign_roi(customer_explanations_full)
roi_df.to_csv(OUTPUTS_DIR / "business_retention_kpis.csv", index=False)
print("Saved business_retention_kpis.csv")

# Strategy quadrant plot
retention_engine.plot_strategy_quadrant(output_path=FIGURES_DIR / "retention_strategy_quadrant.png")
print("Saved retention_strategy_quadrant.png")

# 6. Verification and Summary
print("\n[6/6] Execution Verification Summary:")
print(f"• Total Customers Scored : {len(customer_explanations_full):,}")
print(f"• Low Risk Customers     : {(customer_explanations_full['Risk_Category'] == 'Low Risk').sum():,} ({(customer_explanations_full['Risk_Category'] == 'Low Risk').mean():.1%})")
print(f"• Medium Risk Customers  : {(customer_explanations_full['Risk_Category'] == 'Medium Risk').sum():,} ({(customer_explanations_full['Risk_Category'] == 'Medium Risk').mean():.1%})")
print(f"• High Risk Customers    : {(customer_explanations_full['Risk_Category'] == 'High Risk').sum():,} ({(customer_explanations_full['Risk_Category'] == 'High Risk').mean():.1%})")
print(f"• Top Global Driver      : {df_global_imp.iloc[0]['Feature']} (Odds Ratio: {df_global_imp.iloc[0]['Odds_Ratio']:.2f}x)")
print(f"• Output Files Generated : {len(list(OUTPUTS_DIR.glob('*.csv')))} CSVs")
print(f"• Figures Generated      : {len(list(FIGURES_DIR.glob('*.png')))} PNGs")
print("\nGROUP 5 PIPELINE COMPLETED SUCCESSFULLY!")
