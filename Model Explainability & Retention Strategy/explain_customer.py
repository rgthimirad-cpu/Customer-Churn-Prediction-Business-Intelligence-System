"""
explain_customer.py
===================
Interactive CLI tool to inspect individual customer churn explanations,
risk categories, and tailored retention recommendations.

Usage:
    python3 explain_customer.py --id C0001
    python3 explain_customer.py --index 0
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR / "src"))

from explainers import ChurnModelExplainer
from risk_profiler import CustomerRiskProfiler
from retention_engine import RetentionStrategyEngine
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


def main():
    parser = argparse.ArgumentParser(description="Explain individual customer churn predictions.")
    parser.add_argument("--id", type=str, default="C0001", help="Customer ID (e.g., C0001)")
    parser.add_argument("--index", type=int, default=None, help="Row index (0-7042)")
    args = parser.parse_args()

    project_root = CURRENT_DIR.parent
    data_path = project_root / "Feature Engineering & Statistical Analysis" / "outputs" / "customer_churn_feature_engineered.csv"
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)

    CORE_FEATURES = [
        "Senior Citizen", "Partner", "Dependents", "Tenure Months", "Multiple Lines",
        "Internet Service", "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies", "Contract",
        "Paperless Billing", "Payment Method", "Monthly Charges", "Total Charges",
        "Fiber_Monthly_Risk", "New_High_Spend", "New_Monthly_Customer", "Security_Tech_Bundle"
    ]
    TARGET = "Churn Value"

    X = df[CORE_FEATURES].copy()
    y = df[TARGET].copy()

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scaler", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features)
        ],
        remainder="drop"
    )

    clf = LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipeline.fit(X, y)

    explainer = ChurnModelExplainer(pipeline, feature_names=CORE_FEATURES, background_data=X)
    profiler = CustomerRiskProfiler(explainer)
    retention_engine = RetentionStrategyEngine()

    if args.index is not None:
        idx = args.index
    else:
        # Parse customer ID format like C0001 or C001
        clean_id = args.id.upper().strip()
        num_part = "".join(filter(str.isdigit, clean_id))
        if num_part:
            idx = int(num_part) - 1
        else:
            idx = 0

    if idx < 0 or idx >= len(df):
        print(f"Error: Index {idx} is out of bounds (0 to {len(df)-1}).")
        return

    customer_id = f"C{idx+1:04d}"
    cust_features = X.iloc[[idx]]

    # Explain instance
    explanation = explainer.explain_instance(cust_features, top_n=5)
    prob = explanation["churn_probability"]
    risk_cat = "High Risk" if prob >= 0.60 else ("Medium Risk" if prob >= 0.30 else "Low Risk")

    # Positive and negative factors
    top_pos = explanation["top_positive_factors"]
    top_neg = explanation["top_negative_factors"]

    # Assign retention action
    dummy_row = {
        "Risk_Category": risk_cat,
        "Primary_Churn_Driver": top_pos[0]["Feature"] if top_pos else "None",
        "Secondary_Churn_Driver": top_pos[1]["Feature"] if len(top_pos) > 1 else "None",
        "All_Top_Drivers": ", ".join([f["Feature"] for f in top_pos])
    }
    action = retention_engine.assign_action_to_customer(dummy_row)

    print("\n" + "=" * 65)
    print(f"CUSTOMER EXPLAINABILITY & RETENTION CARD: {customer_id}")
    print("=" * 65)
    print(f"• Churn Probability   : {prob:.1%}")
    print(f"• Risk Tier           : {risk_cat}")
    print(f"• Baseline Expectation: {explanation['base_value_prob']:.1%}")
    print(f"• Actual Status in Data: {'Churned (1)' if y.iloc[idx] == 1 else 'Retained (0)'}")

    print("\n[+] Top Risk Factors (Increasing Churn Likelihood):")
    if top_pos:
        for f in top_pos:
            print(f"   ▲ {f['Feature']:<30} (SHAP impact: {f['SHAP_Value']:+.3f})")
    else:
        print("   None (All factors protective)")

    print("\n[-] Top Mitigating Factors (Reducing Churn Likelihood):")
    if top_neg:
        for f in top_neg:
            print(f"   ▼ {f['Feature']:<30} (SHAP impact: {f['SHAP_Value']:+.3f})")
    else:
        print("   None (All factors increase risk)")

    print("\n🎯 RECOMMENDED RETENTION ACTION:")
    print(f"• Action Play         : {action['Action_Name']}")
    print(f"• Recommended Tactic  : {action['Recommended_Action']}")
    print(f"• Engagement Channel  : {action['Engagement_Channel']}")
    print(f"• Operational Urgency : {action['Action_Urgency']}")
    print(f"• Expected Impact     : {action['Expected_Retention_Lift']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
