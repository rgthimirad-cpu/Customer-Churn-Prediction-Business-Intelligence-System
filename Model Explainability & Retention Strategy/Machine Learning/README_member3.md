# Member 3 — Random Forest & Class Imbalance Analysis

**Branch:** `member3-ml-random-forest` | **Base:** `group4-machine-learning`

## Contents
```
Machine Learning/
├── notebooks/
│   └── 03_random_forest.ipynb          # Full, executed Member 3 notebook
├── outputs/
│   ├── member3_random_forest_cv_results.csv
│   ├── member3_random_forest_tuning_results.csv
│   ├── member3_random_forest_imbalance_results.csv
│   └── member3_random_forest_feature_importance.csv
├── figures/
│   ├── member3_random_forest_confusion_matrix.png
│   ├── member3_random_forest_roc_curve.png
│   ├── member3_random_forest_imbalance_comparison.png
│   └── member3_random_forest_feature_importance.png
└── models/
    └── member3_random_forest_candidate.joblib
```

## How to run
1. Place the Group 3 handoff file at:
   `Feature Engineering & Statistical Analysis/outputs/customer_churn_feature_engineered.csv`
2. Open `notebooks/03_random_forest.ipynb` and run top-to-bottom (Kernel → Restart & Run All).
3. Outputs/figures/model regenerate into the paths above automatically.

## Data note (important — read before merging)
This run used **`customer_churn_cleaned.xlsx` (`Model_Ready_Data` sheet)** — 7,043 rows, 73.46% /
26.54% class split, matching the documented Group 3 handoff numbers. It contains the **17 of 21**
documented core features; the four engineered columns (`Fiber_Monthly_Risk`, `New_High_Spend`,
`New_Monthly_Customer`, `Security_Tech_Bundle`) were not present in this file and so are **not** used.
`Zip Code`, `Total Frequency`, `CLTV`, `Gender`, and `Phone Service` were dropped per the protocol
(redundant / near-perfect-correlation / low-priority). No leakage columns (`Churn Label`, `Churn
Score`, `Churn Reason`) were present or used.

**Action for the team:** once Group 3's fully engineered CSV (with the 4 extra columns) is merged into
the repo at the path above, re-run this notebook unchanged — same protocol, same `random_state=42` — to
confirm the imbalance recommendation and tuned metrics still hold.

## Headline results (this run)
- Best imbalance strategy by CV F1: **natural distribution** (class weighting and SMOTE did not
  improve F1 over the natural class distribution on this feature set — see
  `member3_random_forest_imbalance_results.csv` and the imbalance comparison figure for full evidence).
- Tuned Random Forest CV (5-fold): Accuracy ≈0.81, Precision ≈0.68, Recall ≈0.54, F1 ≈0.60,
  ROC-AUC ≈0.86.
- Full held-out test metrics, confusion matrix, and ROC curve are in the notebook (Section 11) and
  `figures/`.

## What's NOT done here (by design, per protocol)
- Final model selection across all four branches (Logistic Regression / Decision Tree / Random Forest /
  XGBoost) — decided together after integration.
- Classification threshold tuning — reserved for a shared, final-integration decision.
- Any change to the shared split / CV / preprocessing protocol.
