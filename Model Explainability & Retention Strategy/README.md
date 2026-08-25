<<<<<<< HEAD
# Group 5: Model Explainability & Retention Strategy

**Customer Churn Prediction & Business Intelligence System**  
**Team Size:** 3 Members  
**Main Objective:** Explain why customers are likely to churn and convert model findings into practical, ROI-positive retention actions.

---

## Executive Summary

Group 5 bridges the gap between statistical machine learning predictions and executive business decision-making. By leveraging model coefficients, Odds Ratios ($OR = e^{\beta}$), and exact Shapley Additive Explanations (SHAP), we provide transparent global and local explainability for the final deployment model (**Tuned Logistic Regression**). 

Furthermore, we stratify the customer base into actionable **Low**, **Medium**, and **High** risk categories, attribute individual prediction drivers to each customer, and formulate a targeted **Churn Driver → Customer Segment → Recommended Action** retention framework that generates an estimated **$1,770,590 USD** in preserved customer lifetime value (CLTV) with an ROI of **1,070%**.

---

## 1. Directory Structure

```text
Model Explainability & Retention Strategy/
├── README.md                                  <- Master documentation for Group 5
├── explainability_and_retention.ipynb          <- Integrated master notebook
├── generate_outputs.py                        <- Master pipeline execution script
├── notebooks/
│   ├── 01_global_feature_importance_shap.ipynb <- Member 1: Coefficients, Odds Ratios, Global SHAP
│   ├── 02_local_explanations_risk_tiers.ipynb  <- Member 2: Local explanations, waterfalls, risk tiers
│   └── 03_retention_strategy_matrix.ipynb      <- Member 3: Action matrix, ROI simulation, business playbook
├── src/
│   ├── __init__.py
│   ├── explainers.py                          <- SHAP & linear coefficient explainability engine
│   ├── risk_profiler.py                       <- Risk stratification & factor attribution engine
│   └── retention_engine.py                    <- Automated recommendation & ROI calculator
├── outputs/
│   ├── global_feature_importance.csv          <- Model coefficients, odds ratios, and directional effects
│   ├── global_shap_summary.csv                <- Mean absolute SHAP values and importance rankings
│   ├── customer_risk_summary.csv              <- Distribution of customers across Low, Medium, High risk
│   ├── high_risk_profile_analysis.csv         <- Statistical prevalence of drivers within high-risk cohort
│   ├── churn_driver_action_matrix.csv         <- Operational retention strategy matrix
│   ├── customer_level_explanations.csv        <- All 7,043 scored customers with probability, drivers & action
│   └── business_retention_kpis.csv            <- Financial simulation of retention program ROI
└── figures/
    ├── global_feature_importance_odds_ratios.png
    ├── shap_global_bar.png
    ├── shap_summary_beeswarm.png
    ├── risk_distribution_donut.png
    ├── high_risk_drivers_breakdown.png
    ├── individual_waterfall_c0001.png
    ├── individual_waterfall_c0002.png
    ├── individual_waterfall_c0007.png
    └── retention_strategy_quadrant.png
```

---

## 2. Team Member Responsibilities & Deliverables

| Member | Focus Area | Core Responsibilities | Key Deliverables |
|---|---|---|---|
| **Member 1** | **Global Explainability & SHAP Analysis** | Extract standardized model coefficients, compute Odds Ratios ($OR = e^{\beta}$), fit global SHAP explainers, analyze beeswarm plots and interaction dependencies, categorize churn accelerators vs protectors. | • `01_global_feature_importance_shap.ipynb`<br>• `global_feature_importance.csv`<br>• `global_shap_summary.csv`<br>• SHAP summary & beeswarm plots |
| **Member 2** | **Local Explainability & Customer Risk Tiers** | Build local instance explainers, generate individual waterfall decompositions, establish Low/Medium/High risk categories, profile key factors driving high-risk predictions. | • `02_local_explanations_risk_tiers.ipynb`<br>• `customer_risk_summary.csv`<br>• `high_risk_profile_analysis.csv`<br>• Customer waterfall charts |
| **Member 3** | **Retention Strategy & Business Playbook** | Construct the Churn Driver $\rightarrow$ Customer Segment $\rightarrow$ Recommended Action matrix, assign individualized retention plays, model retention campaign financials & ROI, prepare dashboard handoffs. | • `03_retention_strategy_matrix.ipynb`<br>• `churn_driver_action_matrix.csv`<br>• `customer_level_explanations.csv`<br>• `business_retention_kpis.csv`<br>• Strategic priority quadrant |

---

## 3. Global Feature Importance & SHAP Analysis

### 3.1 Model Coefficients & Odds Ratios ($OR = e^{\beta}$)

The final tuned logistic regression model maps input features $X$ to churn log-odds:
$$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{j=1}^{K} \beta_j X_j$$

For a unit increase in standardized feature $X_j$, the odds of churn change by a factor of $e^{\beta_j}$.

| Rank | Feature | Standardized Coefficient ($\beta$) | Odds Ratio ($OR$) | Churn Multiplier ($\%$) | Direction & Classification |
|:---:|---|:---:|:---:|:---:|---|
| **1** | `Internet Service_No` | -0.9903 | 0.3715 | -62.85% | **Protector** (Reduces Churn) |
| **2** | `Internet Service_Fiber optic` | +0.9145 | 2.4956 | +149.56% | **Accelerator** (Increases Churn) |
| **3** | `Contract_Two year` | -0.7351 | 0.4795 | -52.05% | **Protector** (Reduces Churn) |
| **4** | `Contract_Month-to-month` | +0.7060 | 2.0258 | +102.58% | **Accelerator** (Increases Churn) |
| **5** | `Tenure Months` | -0.6551 | 0.5194 | -48.06% | **Protector** (Reduces Churn) |
| **6** | `Dependents` | -0.6380 | 0.5283 | -47.17% | **Protector** (Reduces Churn) |
| **7** | `New_Monthly_Customer` | +0.3115 | 1.3654 | +36.54% | **Accelerator** (Increases Churn) |
| **8** | `Monthly Charges` | -0.1987 | 0.8198 | -18.02% | **Protector** (Reduces Churn, controlling for Fiber) |
| **9** | `Streaming Movies` | +0.1772 | 1.1939 | +19.39% | **Accelerator** (Increases Churn) |
| **10** | `Streaming TV` | +0.1764 | 1.1929 | +19.29% | **Accelerator** (Increases Churn) |

### 3.2 Global SHAP Importance

SHAP values allocate credit for the model prediction $f(x)$ across features relative to the base reference expectation $E[f(X)]$. The mean absolute SHAP value measures global feature impact:

$$\text{Mean } |\text{SHAP}_j| = \frac{1}{N}\sum_{i=1}^{N} |\phi_{i,j}|$$

- **Top 5 Most Influential Features by Mean |SHAP|**:
  1. `Tenure Months` (Mean $|\text{SHAP}| = 0.5835$)
  2. `Dependents` (Mean $|\text{SHAP}| = 0.5378$)
  3. `Internet Service_Fiber optic` (Mean $|\text{SHAP}| = 0.4506$)
  4. `Contract_Month-to-month` (Mean $|\text{SHAP}| = 0.3494$)
  5. `Internet Service_No` (Mean $|\text{SHAP}| = 0.3361$)

---

## 4. Customer Risk Categorization & Profiling

### 4.1 Risk Tier Stratification

Calibrated churn probabilities $p_i \in [0, 1]$ are segmented into three operational business tiers:

| Risk Category | Churn Probability Threshold | Customer Count | Base Share (%) | Avg. Monthly Charges | Avg. Tenure (Months) | Actual Churn Rate (%) | Business Posture |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Low Risk** | $p < 0.30$ | 3,217 | 45.68% | $55.75 | 47.00 | 4.54% | Loyalty Nurturing & Up-selling |
| **Medium Risk** | $0.30 \le p < 0.60$ | 1,498 | 21.27% | $65.03 | 28.99 | 24.50% | Proactive Engagement & Nudges |
| **High Risk** | $p \ge 0.60$ | 2,328 | 33.05% | $77.04 | 14.34 | 58.25% | Urgent Personalized Intervention |
| **Total / Overall** | **—** | **7,043** | **100.0%** | **$64.76** | **32.37** | **26.54%** | **Enterprise Portfolio** |

### 4.2 High-Risk Customer Profile Drivers

Among the **2,328 High Risk** customers, factor attribution reveals clear systemic vulnerabilities:
- **74.9%** have `Internet Service_Fiber optic` in their top 3 risk drivers (high monthly spend and sensitivity to service interruptions).
- **74.1%** have short `Tenure Months` in their top 3 risk drivers (early lifecycle vulnerability, onboarding friction).
- **61.7%** have lack of `Dependents` / solo household structure in their top drivers (lower structural switching costs).
- **61.1%** are `New_Monthly_Customer` (short tenure combined with month-to-month flexibility).
- **26.8%** have explicit `Contract_Month-to-month` as a primary driver.

---

## 5. Local Individual Prediction Explanations

Each individual customer prediction is explained through a local additive decomposition:
$$f(x_i) = \phi_0 + \sum_{j=1}^{K} \phi_{i,j}$$

### Exemplar Explanations:

#### Example 1: Customer C0001 (High Risk)
- **Churn Probability:** `66.2%` $\rightarrow$ **High Risk**
- **Top Risk Drivers:** 
  1. `Tenure Months = 2` (SHAP: $+0.837$)
  2. `New_Monthly_Customer = 1` (SHAP: $+0.264$)
  3. `Dependents = 0` (SHAP: $+0.279$)
- **Mitigating Factors:** `Internet Service = DSL` (SHAP: $-0.287$), `Online Security = 1` (SHAP: $-0.134$)
- **Recommended Action:** First-Year Milestone Loyalty Credit + Complimentary 3-Month Speed Upgrade (`ACT_07`)

#### Example 2: Customer C0007 (Critical High Risk)
- **Churn Probability:** `85.2%` $\rightarrow$ **High Risk**
- **Top Risk Drivers:** 
  1. `Tenure Months = 1` (SHAP: $+0.865$)
  2. `Internet Service = Fiber Optic` (SHAP: $+0.596$)
  3. `Contract = Month-to-Month` (SHAP: $+0.428$)
- **Recommended Action:** Proactive VIP Digital Onboarding Concierge + 15% 6-Month Loyalty Discount (`ACT_03`)

---

## 6. Retention Strategy Matrix & Business Playbook

### 6.1 Churn Driver $\rightarrow$ Customer Segment $\rightarrow$ Recommended Action Matrix

| Action ID | Churn Driver Trigger | Target Customer Segment | Recommended Retention Action | Channel | Urgency | Expected Lift | Unit Cost |
|:---:|---|---|---|---|:---:|:---:|:---:|
| **ACT_01** | Month-to-Month Contract | Month-to-Month subscribers with elevated risk | **12/24-Month Long-Term Contract Incentive** with $15/mo discount & streaming perk | In-App / Email | High | 40% Churn Reduction | $60 |
| **ACT_02** | Fiber Optic High Spend | Fiber Optic users paying > $80/month | **Personalized Value Plan Review** + Speed Guarantee + Free Tech Support Bundle | Account Manager / Outbound Call | High | 35% Churn Reduction | $45 |
| **ACT_03** | New Customer High Spend | Early-tenure customers (<12 mo) on month-to-month plans | **Proactive VIP Digital Onboarding Concierge** + 15% 6-Month Loyalty Discount | Concierge Call / SMS | Immediate | 38% Churn Reduction | $50 |
| **ACT_04** | Lack of Tech Support / Security | Broadband users without essential support add-ons | **3-Month Free Trial** of 'CyberSafe & Tech Support' Bundle | Portal / In-App Banner | Medium | 28% Churn Reduction | $25 |
| **ACT_05** | Electronic Check Billing | Manual Electronic Check bill payers | **$5 Recurring Monthly Bill Credit** for enrolling in Credit Card / Direct Debit Auto-Pay | Billing Portal Pop-up | Medium | 22% Churn Reduction | $20 |
| **ACT_06** | Solo / Senior Household | Single-line / solo subscribers without family attachment | **Multi-Line Family Bundle Incentive** + Dedicated Priority Support Line | Direct Mailer / Phone | Medium | 25% Churn Reduction | $30 |
| **ACT_07** | Short Tenure Lifecycle | Customers in their first 6-12 months of service | **First-Year Milestone Loyalty Credit** + Complimentary 3-Month Speed Upgrade | Email / Mobile App | High | 32% Churn Reduction | $35 |
| **ACT_LOYALTY** | Low Risk Profile | Loyal, stable customers ($p < 0.30$) | **Proactive Loyalty Rewards**, Referral Bonus Perks, and Value-Add Upgrades | Newsletter / App | Routine | Maintain Retention | $10 |

---

## 7. Financial ROI Simulation & Business Impact

Using standard telecom customer economics (Average Customer Lifetime Value $\approx \$2,500$ USD):

| Retention Campaign Segment | Eligible Customers | Expected Baseline Churners | Customers Saved (Intervention) | Gross CLTV Preserved (USD) | Campaign Budget (USD) | Net Value Created (USD) | Campaign ROI (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **High Risk Cohort** (Urgent Proactive) | 2,328 | 1,746.0 | 558.7 | $1,396,800.00 | $128,040.00 | $1,268,760.00 | **990.9%** |
| **Medium Risk Cohort** (Nudge & Education) | 1,498 | 674.1 | 215.7 | $539,280.00 | $37,450.00 | $501,830.00 | **1,340.0%** |
| **Combined Enterprise Program** | **3,826** | **2,420.1** | **774.4** | **$1,936,080.00** | **$165,490.00** | **$1,770,590.00** | **1,069.9%** |

---

## 8. Handoff to Group 6 (Prototype) and Group 7 (Dashboard)

1. **Scored Dataset with Actions:** `outputs/customer_level_explanations.csv` provides all 7,043 customer records with `Customer_ID`, `Churn_Probability`, `Risk_Category`, `Primary_Churn_Driver`, `Top_Driver_SHAP_Impact`, `Action_ID`, `Action_Name`, `Recommended_Action`, and `Engagement_Channel`.
2. **Strategy Matrix:** `outputs/churn_driver_action_matrix.csv` serves as the reference catalog for dynamic recommendation widgets.
3. **Reusable Python Engine:** `src/explainers.py`, `src/risk_profiler.py`, and `src/retention_engine.py` can be imported directly into interactive Streamlit / Flask applications to explain new customer inputs in real-time.
=======
# Customer Churn Prediction & Business Intelligence System

A group project that builds an end-to-end pipeline for predicting customer churn and turning the results into actionable business intelligence — covering data engineering, exploratory analysis, feature engineering, machine learning, model explainability, a prediction prototype, and a BI dashboard.

## Overview

Customer churn — when a customer stops using a company's product or service — is costly to acquire replacements for. This project analyzes a customer churn dataset to identify which customers are likely to churn, explain *why*, and recommend retention actions, packaged into both a prediction interface and a business intelligence dashboard.

>>>>>>> origin/main
