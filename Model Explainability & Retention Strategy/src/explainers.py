"""
explainers.py
=============
Model Explainability Engine for Group 5.
Provides:
- Exact SHAP computation for Linear/Logistic and Tree models
- Standardized coefficients and Odds Ratios (OR = exp(beta))
- Global Feature Importance and SHAP beeswarm / bar visualizations
- Local instance explanations (waterfall / force-style factor breakdown)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ChurnModelExplainer:
    """
    Comprehensive explainability engine for the Customer Churn Prediction model.
    """

    def __init__(self, pipeline, feature_names=None, background_data=None):
        """
        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            Trained pipeline containing 'preprocessor' and 'classifier'.
        feature_names : list of str, optional
            List of raw input feature names.
        background_data : pd.DataFrame, optional
            Background dataset used for calculating baseline expectations.
        """
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        self.raw_feature_names = feature_names

        # Extract transformed column names
        self.transformed_feature_names = self._extract_transformed_feature_names()

        # Extract coefficients and intercept
        self.coef = self.classifier.coef_[0]
        self.intercept = self.classifier.intercept_[0]

        self.background_data = background_data
        self.background_transformed = None
        self.expected_value_log_odds = None
        self.expected_value_prob = None
        self.feature_means = None

        if background_data is not None:
            self.set_background(background_data)

    def _extract_transformed_feature_names(self):
        """Extract the exact feature names produced by the ColumnTransformer."""
        feature_names = []
        for name, transformer, cols in self.preprocessor.transformers_:
            if name == "remainder" and transformer == "drop":
                continue
            if hasattr(transformer, "named_steps"):
                last_step = list(transformer.named_steps.values())[-1]
                if hasattr(last_step, "get_feature_names_out"):
                    names = last_step.get_feature_names_out(cols).tolist()
                    feature_names.extend(names)
                else:
                    feature_names.extend(cols)
            elif hasattr(transformer, "get_feature_names_out"):
                names = transformer.get_feature_names_out(cols).tolist()
                feature_names.extend(names)
            else:
                feature_names.extend(cols)
        return feature_names

    def set_background(self, background_df):
        """Set the background reference dataset for expectation calculations."""
        self.background_data = background_df
        self.background_transformed = self.preprocessor.transform(background_df)
        self.feature_means = np.mean(self.background_transformed, axis=0)
        # Expected log-odds E[f(X)] = beta_0 + sum(beta_j * E[X_j])
        self.expected_value_log_odds = float(self.intercept + np.dot(self.coef, self.feature_means))
        # Expected probability = sigmoid(E[f(X)])
        self.expected_value_prob = float(1.0 / (1.0 + np.exp(-self.expected_value_log_odds)))

    def transform_data(self, X):
        """Transform raw input DataFrame through the pipeline preprocessor."""
        return self.preprocessor.transform(X)

    def compute_global_feature_importance(self):
        """
        Compute standardized coefficients, Odds Ratios (OR = exp(beta)),
        and driver classifications.
        """
        odds_ratios = np.exp(self.coef)
        pct_change = (odds_ratios - 1.0) * 100.0

        importance_df = pd.DataFrame({
            "Feature": self.transformed_feature_names,
            "Coefficient": self.coef,
            "Absolute_Coefficient": np.abs(self.coef),
            "Odds_Ratio": odds_ratios,
            "Odds_Multiplier_Pct": pct_change,
            "Direction": np.where(self.coef > 0, "Accelerator (Increases Churn)", "Protector (Reduces Churn)")
        })

        importance_df = importance_df.sort_values(by="Absolute_Coefficient", ascending=False).reset_index(drop=True)
        importance_df["Rank"] = range(1, len(importance_df) + 1)
        return importance_df

    def compute_shap_values(self, X):
        """
        Compute exact Linear SHAP values for each feature of instances in X.
        SHAP value for feature j: phi_j(x) = beta_j * (x_j - E[X_j])
        Sum of SHAP values = f(x) - E[f(X)]
        """
        if self.background_transformed is None:
            raise ValueError("Background data must be set before computing SHAP values.")

        X_trans = self.transform_data(X)
        diff = X_trans - self.feature_means  # (N, D)
        shap_values = diff * self.coef       # Element-wise multiplication (N, D)

        return shap_values, self.expected_value_log_odds

    def compute_global_shap_summary(self, X):
        """
        Calculate global mean absolute SHAP value for each feature across dataset X.
        """
        shap_vals, _ = self.compute_shap_values(X)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        shap_summary_df = pd.DataFrame({
            "Feature": self.transformed_feature_names,
            "Mean_Abs_SHAP": mean_abs_shap,
            "Coefficient": self.coef,
            "Direction": np.where(self.coef > 0, "Increases Churn", "Reduces Churn")
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        shap_summary_df["SHAP_Rank"] = range(1, len(shap_summary_df) + 1)
        return shap_summary_df

    def explain_instance(self, X_row, top_n=5):
        """
        Explain a single customer prediction instance.
        Returns prediction probability, log-odds, base value,
        and top positive/negative contributing factors.
        """
        if isinstance(X_row, pd.Series):
            X_df = X_row.to_frame().T
        elif isinstance(X_row, pd.DataFrame):
            X_df = X_row.iloc[[0]]
        else:
            raise TypeError("X_row must be a pandas Series or 1-row DataFrame.")

        # Compute probability and log-odds
        prob = self.pipeline.predict_proba(X_df)[0, 1]
        X_trans = self.transform_data(X_df)
        diff = X_trans - self.feature_means
        shap_vals = (diff * self.coef)[0]
        model_log_odds = float(self.intercept + np.dot(self.coef, X_trans[0]))

        explanation_df = pd.DataFrame({
            "Feature": self.transformed_feature_names,
            "Transformed_Value": X_trans[0],
            "Mean_Value": self.feature_means,
            "SHAP_Value": shap_vals,
            "Absolute_SHAP": np.abs(shap_vals),
            "Impact": np.where(shap_vals > 0, "Increases Risk", "Decreases Risk")
        }).sort_values(by="Absolute_SHAP", ascending=False).reset_index(drop=True)

        top_positive = explanation_df[explanation_df["SHAP_Value"] > 0].head(top_n)
        top_negative = explanation_df[explanation_df["SHAP_Value"] < 0].head(top_n)

        return {
            "churn_probability": float(prob),
            "predicted_log_odds": model_log_odds,
            "base_value_log_odds": self.expected_value_log_odds,
            "base_value_prob": self.expected_value_prob,
            "explanation_table": explanation_df,
            "top_positive_factors": top_positive.to_dict(orient="records"),
            "top_negative_factors": top_negative.to_dict(orient="records")
        }

    # ==========================================
    # Visualizations
    # ==========================================

    def plot_feature_importance(self, top_n=15, output_path=None):
        """Plot standardized logistic coefficients / odds ratios."""
        df_imp = self.compute_global_feature_importance().head(top_n)

        plt.figure(figsize=(10, 6), dpi=300)
        colors = ["#e74c3c" if d == "Accelerator (Increases Churn)" else "#27ae60" for d in df_imp["Direction"]]
        bars = plt.barh(df_imp["Feature"][::-1], df_imp["Coefficient"][::-1], color=colors[::-1], edgecolor="black", alpha=0.85)

        plt.axvline(0, color="black", linestyle="--", linewidth=1.0)
        plt.title(f"Global Feature Importance: Logistic Regression Coefficients (Top {top_n})", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Standardized Logistic Coefficient (Log-Odds Impact)", fontsize=11)
        plt.ylabel("Features", fontsize=11)
        plt.grid(axis="x", linestyle=":", alpha=0.6)

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", edgecolor="black", label="Increases Churn (Accelerator)"),
            Patch(facecolor="#27ae60", edgecolor="black", label="Reduces Churn (Protector)")
        ]
        plt.legend(handles=legend_elements, loc="lower right", frameon=True)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def plot_shap_summary_bar(self, X, top_n=15, output_path=None):
        """Plot Global Mean |SHAP| Value Bar Chart."""
        shap_df = self.compute_global_shap_summary(X).head(top_n)

        plt.figure(figsize=(10, 6), dpi=300)
        bars = plt.barh(shap_df["Feature"][::-1], shap_df["Mean_Abs_SHAP"][::-1], color="#3498db", edgecolor="#2980b9", alpha=0.9)

        plt.title(f"Global SHAP Feature Importance (Mean |SHAP Value| - Top {top_n})", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Mean |SHAP Value| (Average impact on model log-odds)", fontsize=11)
        plt.ylabel("Features", fontsize=11)
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def plot_shap_beeswarm(self, X, top_n=15, output_path=None):
        """
        Plot SHAP Summary Beeswarm plot representing feature value vs SHAP impact.
        """
        shap_vals, _ = self.compute_shap_values(X)
        X_trans = self.transform_data(X)

        # Rank features by mean |SHAP|
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:top_n]

        plt.figure(figsize=(11, 7), dpi=300)

        y_positions = range(top_n)
        for i, idx in enumerate(top_indices[::-1]):
            feature_name = self.transformed_feature_names[idx]
            feature_shaps = shap_vals[:, idx]
            feature_values = X_trans[:, idx]

            # Normalize feature values to [0, 1] for color mapping
            v_min, v_max = np.min(feature_values), np.max(feature_values)
            norm_vals = (feature_values - v_min) / (v_max - v_min + 1e-8)

            # Add jitter for vertical spread
            jitter = np.random.normal(0, 0.08, size=len(feature_shaps))
            scatter = plt.scatter(
                feature_shaps,
                np.full_like(feature_shaps, i) + jitter,
                c=norm_vals,
                cmap="coolwarm",
                alpha=0.4,
                s=16,
                edgecolors="none"
            )

        plt.axvline(0, color="grey", linestyle="--", linewidth=1.0)
        plt.yticks(y_positions, [self.transformed_feature_names[i] for i in top_indices[::-1]], fontsize=10)
        plt.xlabel("SHAP Value (Impact on Model Log-Odds / Churn Likelihood)", fontsize=11)
        plt.title(f"SHAP Summary Beeswarm Plot (Top {top_n} Features)", fontsize=13, fontweight="bold", pad=15)

        cbar = plt.colorbar(scatter, orientation="vertical", pad=0.02, shrink=0.7)
        cbar.set_label("Feature Value (Low → High)", fontsize=10)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])

        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def plot_waterfall(self, X_row, customer_id="Customer", max_display=8, output_path=None):
        """
        Plot local waterfall chart explaining a single customer prediction.
        """
        explanation = self.explain_instance(X_row, top_n=max_display)
        table = explanation["explanation_table"].head(max_display).copy()

        prob = explanation["churn_probability"]
        base_val = explanation["base_value_log_odds"]
        pred_val = explanation["predicted_log_odds"]

        # Sort table so that bars flow sequentially
        table = table.sort_values(by="Absolute_SHAP", ascending=True).reset_index(drop=True)

        features = table["Feature"].tolist()
        shap_values = table["SHAP_Value"].tolist()

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Plot horizontal waterfall bars
        colors = ["#e74c3c" if v > 0 else "#27ae60" for v in shap_values]
        y_pos = range(len(features))

        bars = ax.barh(y_pos, shap_values, color=colors, alpha=0.85, edgecolor="black")

        for idx, bar in enumerate(bars):
            val = shap_values[idx]
            offset = 0.02 if val >= 0 else -0.02
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, idx, f"{val:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.axvline(0, color="black", linestyle="-", linewidth=1.0)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        title = f"Local Explanation for {customer_id} | Churn Prob: {prob:.1%} (Log-Odds: {pred_val:+.2f} vs Base: {base_val:+.2f})"
        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("SHAP Impact on Log-Odds (+ Pushes Toward Churn, - Pushes Toward Retention)", fontsize=10)

        plt.tight_layout()
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
