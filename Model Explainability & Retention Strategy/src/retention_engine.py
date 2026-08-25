"""
retention_engine.py
===================
Actionable Retention Strategy Engine for Group 5.
Implements:
- Churn Driver -> Customer Segment -> Recommended Action Matrix
- Personalized customer-level recommendation assignment
- Financial ROI & Campaign Simulation Model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class RetentionStrategyEngine:
    """
    Translates model explainability outputs and risk tiers into actionable business retention strategies.
    """

    STRATEGY_RULES = [
        {
            "id": "ACT_01",
            "driver_pattern": "Contract_Month-to-month",
            "driver_name": "Month-to-Month Contract",
            "customer_segment": "Month-to-Month Subscribers with High Churn Risk",
            "recommended_action": "12/24-Month Long-Term Contract Incentive with $15/mo discount & streaming perk",
            "channel": "In-App Notification / Email Campaign",
            "urgency": "High",
            "expected_churn_reduction_pct": 40.0,
            "cost_per_customer": 60.0
        },
        {
            "id": "ACT_02",
            "driver_pattern": "Internet Service_Fiber optic",
            "driver_name": "Fiber Optic High Spend / Service Friction",
            "customer_segment": "Fiber Optic Users with Elevated Churn Probability",
            "recommended_action": "Personalized Value Plan Review + Speed Guarantee + Bundled Free Tech Support",
            "channel": "Dedicated Account Specialist / Outbound Call",
            "urgency": "High",
            "expected_churn_reduction_pct": 35.0,
            "cost_per_customer": 45.0
        },
        {
            "id": "ACT_03",
            "driver_pattern": "New_Monthly_Customer",
            "driver_name": "New / Short-Tenure Customer Risk",
            "customer_segment": "Early-Tenure Customers (<12 Months) on Month-to-Month Plans",
            "recommended_action": "Proactive VIP Digital Onboarding Concierge + 15% 6-Month Loyalty Discount",
            "channel": "Personalized Welcome SMS / Concierge Call",
            "urgency": "Immediate",
            "expected_churn_reduction_pct": 38.0,
            "cost_per_customer": 50.0
        },
        {
            "id": "ACT_04",
            "driver_pattern": "Security_Tech_Bundle",
            "driver_name": "Lack of Tech Support & Online Security",
            "customer_segment": "Broadband Users without Essential Support/Security Add-ons",
            "recommended_action": "3-Month Free Trial of 'CyberSafe & Premium Tech Support' Bundle",
            "channel": "Self-Service Portal / Email Prompt",
            "urgency": "Medium",
            "expected_churn_reduction_pct": 28.0,
            "cost_per_customer": 25.0
        },
        {
            "id": "ACT_05",
            "driver_pattern": "Payment Method",
            "driver_name": "Electronic Check / Billing Friction",
            "customer_segment": "Manual Electronic Check Bill Payers",
            "recommended_action": "$5 Recurring Monthly Bill Credit for Enrolling in Auto-Pay via Bank/Credit Card",
            "channel": "Billing Portal Pop-up / SMS Reminder",
            "urgency": "Medium",
            "expected_churn_reduction_pct": 22.0,
            "cost_per_customer": 20.0
        },
        {
            "id": "ACT_06",
            "driver_pattern": "Dependents",
            "driver_name": "Solo Account / Low Attachment Risk",
            "customer_segment": "Single-Line / Solo Subscribers without Family Attachment",
            "recommended_action": "Multi-Line Family Bundle Incentive + Dedicated Customer Care Line",
            "channel": "Direct Mailer / Phone Call",
            "urgency": "Medium",
            "expected_churn_reduction_pct": 25.0,
            "cost_per_customer": 30.0
        },
        {
            "id": "ACT_07",
            "driver_pattern": "Tenure Months",
            "driver_name": "Low Tenure / Early Lifecycle Risk",
            "customer_segment": "Customers in First 6-12 Months of Service",
            "recommended_action": "First-Year Milestone Loyalty Credit + Complimentary 3-Month Speed Upgrade",
            "channel": "Email / App Banner",
            "urgency": "High",
            "expected_churn_reduction_pct": 32.0,
            "cost_per_customer": 35.0
        },
        {
            "id": "ACT_DEFAULT",
            "driver_pattern": "Default",
            "driver_name": "General Retention Risk",
            "customer_segment": "General Elevated Churn Risk Customers",
            "recommended_action": "Personalized Customer Satisfaction Review + Exclusive 10% Service Voucher",
            "channel": "Email / App Banner",
            "urgency": "Standard",
            "expected_churn_reduction_pct": 20.0,
            "cost_per_customer": 25.0
        }
    ]

    def __init__(self):
        self.strategy_matrix_df = pd.DataFrame(self.STRATEGY_RULES)

    def get_strategy_matrix(self):
        """Return the structured retention strategy matrix as a DataFrame."""
        return self.strategy_matrix_df.copy()

    def assign_action_to_customer(self, row):
        """
        Assign specific recommended retention action based on customer risk tier and primary driver.
        """
        risk = row["Risk_Category"]
        if risk == "Low Risk":
            return {
                "Action_ID": "ACT_LOYALTY",
                "Action_Name": "Loyalty Appreciation & Cross-Sell",
                "Recommended_Action": "Proactive Loyalty Rewards, Referral Bonus Perks, and Value-Add Service Upgrades",
                "Engagement_Channel": "Quarterly Newsletter / Digital App",
                "Action_Urgency": "Low / Routine",
                "Expected_Retention_Lift": "Maintain High Retention"
            }

        primary_driver = str(row.get("Primary_Churn_Driver", ""))
        secondary_driver = str(row.get("Secondary_Churn_Driver", ""))
        all_drivers = str(row.get("All_Top_Drivers", ""))

        # Check primary driver first, then secondary, then all top drivers
        for rule in self.STRATEGY_RULES[:-1]:
            pat = rule["driver_pattern"]
            if pat in primary_driver:
                return {
                    "Action_ID": rule["id"],
                    "Action_Name": rule["driver_name"],
                    "Recommended_Action": rule["recommended_action"],
                    "Engagement_Channel": rule["channel"],
                    "Action_Urgency": rule["urgency"] if risk == "High Risk" else "Medium",
                    "Expected_Retention_Lift": f"{rule['expected_churn_reduction_pct']}% Churn Reduction"
                }

        for rule in self.STRATEGY_RULES[:-1]:
            pat = rule["driver_pattern"]
            if pat in secondary_driver or pat in all_drivers:
                return {
                    "Action_ID": rule["id"],
                    "Action_Name": rule["driver_name"],
                    "Recommended_Action": rule["recommended_action"],
                    "Engagement_Channel": rule["channel"],
                    "Action_Urgency": rule["urgency"] if risk == "High Risk" else "Medium",
                    "Expected_Retention_Lift": f"{rule['expected_churn_reduction_pct']}% Churn Reduction"
                }

        default_rule = self.STRATEGY_RULES[-1]
        return {
            "Action_ID": default_rule["id"],
            "Action_Name": default_rule["driver_name"],
            "Recommended_Action": default_rule["recommended_action"],
            "Engagement_Channel": default_rule["channel"],
            "Action_Urgency": "Medium",
            "Expected_Retention_Lift": f"{default_rule['expected_churn_reduction_pct']}% Churn Reduction"
        }

    def enhance_customer_explanations(self, scored_df):
        """
        Enrich scored customer table with recommended retention actions and operational priorities.
        """
        actions = []
        for _, row in scored_df.iterrows():
            act = self.assign_action_to_customer(row)
            actions.append(act)

        act_df = pd.DataFrame(actions)
        enhanced_df = pd.concat([scored_df.reset_index(drop=True), act_df], axis=1)
        return enhanced_df

    def simulate_retention_campaign_roi(self, enhanced_df, avg_customer_cltv=2500.0):
        """
        Simulate financial impact and return on investment (ROI) for targeted retention interventions.
        """
        high_risk = enhanced_df[enhanced_df["Risk_Category"] == "High Risk"]
        med_risk = enhanced_df[enhanced_df["Risk_Category"] == "Medium Risk"]

        def calc_cohort_roi(cohort_df, cohort_name, baseline_retention_rate, cost_per_cust=45.0):
            cust_count = len(cohort_df)
            expected_churners = cust_count * (1.0 - baseline_retention_rate)
            # Assume intervention saves 32% of would-be churners
            saved_customers = expected_churners * 0.32
            gross_value_saved = saved_customers * avg_customer_cltv
            total_campaign_cost = cust_count * cost_per_cust
            net_profit_saved = gross_value_saved - total_campaign_cost
            roi_pct = (net_profit_saved / total_campaign_cost * 100.0) if total_campaign_cost > 0 else 0.0

            return {
                "Segment": cohort_name,
                "Customer_Count": cust_count,
                "Expected_Baseline_Churners": round(expected_churners, 1),
                "Customers_Saved_By_Campaign": round(saved_customers, 1),
                "Gross_CLTV_Preserved_USD": round(gross_value_saved, 2),
                "Retention_Campaign_Cost_USD": round(total_campaign_cost, 2),
                "Net_Value_Created_USD": round(net_profit_saved, 2),
                "Campaign_ROI_Pct": round(roi_pct, 2)
            }

        high_roi = calc_cohort_roi(high_risk, "High Risk Cohort (Urgent Proactive)", baseline_retention_rate=0.25, cost_per_cust=55.0)
        med_roi = calc_cohort_roi(med_risk, "Medium Risk Cohort (Nudge & Education)", baseline_retention_rate=0.55, cost_per_cust=25.0)

        total_count = high_roi["Customer_Count"] + med_roi["Customer_Count"]
        total_saved = high_roi["Customers_Saved_By_Campaign"] + med_roi["Customers_Saved_By_Campaign"]
        total_gross = high_roi["Gross_CLTV_Preserved_USD"] + med_roi["Gross_CLTV_Preserved_USD"]
        total_cost = high_roi["Retention_Campaign_Cost_USD"] + med_roi["Retention_Campaign_Cost_USD"]
        total_net = total_gross - total_cost
        total_roi = (total_net / total_cost * 100.0) if total_cost > 0 else 0.0

        total_row = {
            "Segment": "Combined Retention Program (Total)",
            "Customer_Count": total_count,
            "Expected_Baseline_Churners": round(high_roi["Expected_Baseline_Churners"] + med_roi["Expected_Baseline_Churners"], 1),
            "Customers_Saved_By_Campaign": round(total_saved, 1),
            "Gross_CLTV_Preserved_USD": round(total_gross, 2),
            "Retention_Campaign_Cost_USD": round(total_cost, 2),
            "Net_Value_Created_USD": round(total_net, 2),
            "Campaign_ROI_Pct": round(total_roi, 2)
        }

        roi_df = pd.DataFrame([high_roi, med_roi, total_row])
        return roi_df

    def plot_strategy_quadrant(self, output_path=None):
        """Plot Churn Risk vs CLTV / Effort Strategic Priority Matrix."""
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

        # Plot 4 Quadrants
        ax.axvline(0.5, color="grey", linestyle="--", alpha=0.7)
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.7)

        # Quadrant 1: High Risk, High CLTV (Top Right)
        ax.fill_between([0.5, 1.0], 0.5, 1.0, color="#e74c3c", alpha=0.15)
        ax.text(0.75, 0.75, "URGENT INTERVENTION\nHigh Value, High Risk\n• Dedicated Account Calls\n• Contract Discount Lock-in\n• VIP Tech Support",
                ha="center", va="center", fontsize=10, fontweight="bold", color="#c0392b")

        # Quadrant 2: Low Risk, High CLTV (Top Left)
        ax.fill_between([0.0, 0.5], 0.5, 1.0, color="#2ecc71", alpha=0.15)
        ax.text(0.25, 0.75, "NURTURE & ADVOCATE\nHigh Value, Low Risk\n• Loyalty Rewards\n• Beta Feature Access\n• Referral Incentives",
                ha="center", va="center", fontsize=10, fontweight="bold", color="#27ae60")

        # Quadrant 3: High Risk, Low CLTV (Bottom Right)
        ax.fill_between([0.5, 1.0], 0.0, 0.5, color="#f39c12", alpha=0.15)
        ax.text(0.75, 0.25, "AUTOMATED WIN-BACK\nLow Value, High Risk\n• Automated Email Prompts\n• $5 Auto-Pay Credit\n• Self-Service Bundle",
                ha="center", va="center", fontsize=10, fontweight="bold", color="#d35400")

        # Quadrant 4: Low Risk, Low CLTV (Bottom Left)
        ax.fill_between([0.0, 0.5], 0.0, 0.5, color="#3498db", alpha=0.15)
        ax.text(0.25, 0.25, "CROSS-SELL & GROW\nLow Value, Low Risk\n• Streaming Add-on Upsells\n• Speed Tier Upgrades\n• Device Protection Promos",
                ha="center", va="center", fontsize=10, fontweight="bold", color="#2980b9")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Churn Risk Likelihood (Low → High)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Customer Lifetime Value / Spend (Low → High)", fontsize=11, fontweight="bold")
        ax.set_title("Customer Retention Strategy Matrix & Prioritization Quadrants", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
