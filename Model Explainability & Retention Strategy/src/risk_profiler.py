"""
risk_profiler.py
================
Customer Risk Stratification and Factor Attribution Engine.
Assigns risk categories (Low, Medium, High) and extracts personalized drivers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class CustomerRiskProfiler:
    """
    Categorizes customers by risk tier and profiles key drivers behind high-risk predictions.
    """

    LOW_THRESHOLD = 0.30
    HIGH_THRESHOLD = 0.60

    def __init__(self, explainer):
        """
        Parameters
        ----------
        explainer : ChurnModelExplainer
            Configured explainer instance.
        """
        self.explainer = explainer

    def categorize_risk(self, probabilities):
        """
        Assign risk categories based on calibrated churn probabilities:
        - Low Risk: prob < 0.30
        - Medium Risk: 0.30 <= prob < 0.60
        - High Risk: prob >= 0.60
        """
        categories = pd.cut(
            probabilities,
            bins=[-0.001, self.LOW_THRESHOLD, self.HIGH_THRESHOLD, 1.001],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )
        return categories

    def score_and_profile_dataset(self, df_raw, features, customer_id_col=None):
        """
        Score the entire dataset and generate individual explanations, risk tiers,
        and top 3 risk factors.
        """
        X = df_raw[features].copy()
        probs = self.explainer.pipeline.predict_proba(X)[:, 1]
        preds = (probs >= 0.50).astype(int)
        risk_tiers = self.categorize_risk(probs)

        # Compute SHAP values for all rows
        shap_vals, base_val = self.explainer.compute_shap_values(X)
        feat_names = self.explainer.transformed_feature_names

        # Assign Customer IDs if not present
        if customer_id_col and customer_id_col in df_raw.columns:
            cust_ids = df_raw[customer_id_col].values
        else:
            cust_ids = [f"C{i+1:04d}" for i in range(len(df_raw))]

        records = []
        for i in range(len(df_raw)):
            row_shaps = shap_vals[i]
            # Find top positive (risk-increasing) and top negative (protective) features
            pos_indices = np.where(row_shaps > 0)[0]
            neg_indices = np.where(row_shaps < 0)[0]

            # Sort positive indices descending
            pos_sorted = pos_indices[np.argsort(-row_shaps[pos_indices])]
            neg_sorted = neg_indices[np.argsort(row_shaps[neg_indices])]

            top_pos_names = [feat_names[idx] for idx in pos_sorted[:3]]
            top_pos_scores = [round(row_shaps[idx], 3) for idx in pos_sorted[:3]]

            top_neg_names = [feat_names[idx] for idx in neg_sorted[:2]]
            top_neg_scores = [round(row_shaps[idx], 3) for idx in neg_sorted[:2]]

            main_drivers_str = ", ".join(top_pos_names) if top_pos_names else "None"
            mitigating_str = ", ".join(top_neg_names) if top_neg_names else "None"

            records.append({
                "Customer_ID": cust_ids[i],
                "Churn_Probability": round(float(probs[i]), 4),
                "Predicted_Churn": int(preds[i]),
                "Risk_Category": str(risk_tiers[i]),
                "Primary_Churn_Driver": top_pos_names[0] if top_pos_names else "None",
                "Secondary_Churn_Driver": top_pos_names[1] if len(top_pos_names) > 1 else "None",
                "Tertiary_Churn_Driver": top_pos_names[2] if len(top_pos_names) > 2 else "None",
                "All_Top_Drivers": main_drivers_str,
                "Mitigating_Factors": mitigating_str,
                "Top_Driver_SHAP_Impact": top_pos_scores[0] if top_pos_scores else 0.0
            })

        scored_df = pd.DataFrame(records)
        return scored_df

    def compute_risk_summary(self, scored_df, df_original=None):
        """
        Generate statistical breakdown across Low, Medium, and High Risk segments.
        """
        summary = scored_df.groupby("Risk_Category", observed=False).agg(
            Customer_Count=("Customer_ID", "count"),
            Mean_Churn_Prob=("Churn_Probability", "mean"),
            Min_Churn_Prob=("Churn_Probability", "min"),
            Max_Churn_Prob=("Churn_Probability", "max")
        ).reset_index()

        total_cust = len(scored_df)
        summary["Percentage"] = (summary["Customer_Count"] / total_cust * 100.0).round(2)
        summary["Mean_Churn_Prob"] = (summary["Mean_Churn_Prob"] * 100.0).round(2)

        # Merge with business features if original dataset is provided
        if df_original is not None:
            merged = pd.concat([scored_df[["Risk_Category"]], df_original], axis=1)
            metric_cols = {}
            if "Monthly Charges" in merged.columns:
                metric_cols["Avg_Monthly_Charges"] = ("Monthly Charges", "mean")
            if "Tenure Months" in merged.columns:
                metric_cols["Avg_Tenure_Months"] = ("Tenure Months", "mean")
            if "Total Charges" in merged.columns:
                metric_cols["Avg_Total_Charges"] = ("Total Charges", "mean")
            if "Churn Value" in merged.columns:
                metric_cols["Actual_Churn_Rate"] = ("Churn Value", lambda x: np.mean(x) * 100.0)

            if metric_cols:
                extra_summary = merged.groupby("Risk_Category", observed=False).agg(**metric_cols).reset_index()
                summary = summary.merge(extra_summary, on="Risk_Category", how="left").round(2)

        return summary

    def analyze_high_risk_factors(self, scored_df):
        """
        Aggregate the most frequent primary and secondary churn drivers among High Risk customers.
        """
        high_risk_df = scored_df[scored_df["Risk_Category"] == "High Risk"]
        total_high_risk = len(high_risk_df)

        primary_counts = high_risk_df["Primary_Churn_Driver"].value_counts().reset_index()
        primary_counts.columns = ["Driver", "Primary_Count"]

        all_drivers_series = pd.concat([
            high_risk_df["Primary_Churn_Driver"],
            high_risk_df["Secondary_Churn_Driver"],
            high_risk_df["Tertiary_Churn_Driver"]
        ])
        all_counts = all_drivers_series[all_drivers_series != "None"].value_counts().reset_index()
        all_counts.columns = ["Driver", "Total_Occurrences_In_Top3"]

        factor_profile = primary_counts.merge(all_counts, on="Driver", how="outer").fillna(0)
        factor_profile["High_Risk_Prevalence_Pct"] = (factor_profile["Total_Occurrences_In_Top3"] / total_high_risk * 100.0).round(2)
        factor_profile = factor_profile.sort_values(by="Total_Occurrences_In_Top3", ascending=False).reset_index(drop=True)

        return factor_profile

    def plot_risk_distribution(self, summary_df, output_path=None):
        """Plot donut chart of Customer Risk Categories."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        labels = [f"{row['Risk_Category']}\n({row['Customer_Count']:,} cust | {row['Percentage']}%)" for _, row in summary_df.iterrows()]

        wedges, texts, autotexts = ax.pie(
            summary_df["Customer_Count"],
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.75,
            textprops={"fontsize": 10, "fontweight": "bold"},
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
        )

        plt.title("Customer Distribution by Churn Risk Category", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def plot_high_risk_drivers(self, factor_profile, top_n=10, output_path=None):
        """Plot horizontal bar chart of top drivers within high-risk cohort."""
        top_factors = factor_profile.head(top_n)

        plt.figure(figsize=(10, 6), dpi=300)
        bars = plt.barh(
            top_factors["Driver"][::-1],
            top_factors["High_Risk_Prevalence_Pct"][::-1],
            color="#e74c3c",
            edgecolor="#c0392b",
            alpha=0.85
        )

        plt.title(f"Main Factors Driving High-Risk Customer Predictions (% of High-Risk Segment)", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("% of High-Risk Customers with Factor in Top 3 Drivers", fontsize=11)
        plt.ylabel("Risk Driver", fontsize=11)
        plt.grid(axis="x", linestyle=":", alpha=0.6)

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1.0, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=9, fontweight="bold")

        plt.xlim(0, max(top_factors["High_Risk_Prevalence_Pct"]) + 12)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
