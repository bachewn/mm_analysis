#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
March Madness First-Round Win Classifier
----------------------------------------
Builds and evaluates models to predict Team_Won from pace, seed, and efficiency metrics.

Inputs
------
- Excel file with columns:
  TeamName, OpponentName, WinnerName, Team_Pace, Opp_Pace, Team_Seed, Opp_Seed,
  NET_Diff, Team_NET, Opp_NET, Team_Turnovers, Opp_Turnovers, Team_EFG, Opp_EFG,
  Team_FTA, Opp_FTA, Team_Rebounds, Rebounds_Diff, Team_Won, Net_OE, Team_OE, Opp_OE

Target
------
- Team_Won (1 if that team won, else 0)

Models
------
- Logistic Regression (with scaling)
- Random Forest
- Gradient Boosting (XGBoost-like via sklearn's GradientBoostingClassifier)

Outputs
-------
- Console summaries
- Plots and artifacts in ./outputs:
    * correlation heatmap
    * distribution plots
    * ROC curves
    * confusion matrices
    * feature importances (tree-based)
    * SHAP summary plots (if available)
    * coefficients / odds ratios (logistic)
    * a CSV of test set with predicted probabilities

Notes
-----
- Ensures reproducibility via a fixed random_state.
- Uses only the requested predictors: Team_Pace, Opp_Pace, Team_Seed, Opp_Seed,
  Team_NET, Opp_NET, Team_OE, Opp_OE
- Also constructs informative engineered diffs (pace_diff, seed_diff, net_diff, oe_diff),
  but keeps the requested base features in the model. Engineered features are used
  for EDA and are optional for modeling (toggled by --use_diffs).

Author: (you)
"""

import argparse
import os
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import os
os.environ["MPLBACKEND"] = "Agg"  # must be set before importing pyplot

import matplotlib
matplotlib.use("Agg")             # belt & suspenders: force headless backend

import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

# Optional: SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with `pip install shap` for SHAP plots.", RuntimeWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def plot_odds_ratios(coef_csv_path: str, out_path: str = "outputs/odds_ratio_plot.png"):
    """
    Plot percent change in win odds per 1 SD change in each feature.
    """
    df = pd.read_csv(coef_csv_path)
    df["pct_change"] = (df["odds_ratio"] - 1.0) * 100.0  # +% or -%
    df = df.sort_values("pct_change", ascending=True)

    plt.figure(figsize=(8, 6))
    colors = df["pct_change"].apply(lambda x: "green" if x > 0 else "red")
    plt.barh(df["feature"], df["pct_change"], color=colors)
    plt.axvline(0, color="k", lw=1)
    plt.title("% Change in Win Odds per 1 SD of Feature")
    plt.xlabel("Δ Win Odds (%) per 1 SD")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[Viz] Odds ratio plot saved to: {os.path.abspath(out_path)}")

def plot_net_diff_curve(model, feature_cols, out_path: str = "outputs/net_diff_win_prob_curve.png"):
    """
    Plot predicted win probability vs NET difference, holding other features at mean.
    Assumes model is logistic regression pipeline.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # Build input grid
    net_range = np.linspace(-20, 20, 200)
    X_base = pd.DataFrame(0, index=np.arange(len(net_range)), columns=feature_cols)

    # Find which column is net_diff and fill range
    if "net_diff" not in X_base.columns:
        raise ValueError("net_diff not found among model features.")

    X_base["net_diff"] = net_range

    # Predict
    p = model.predict_proba(X_base)[:, 1]

    plt.figure(figsize=(7, 5))
    plt.plot(net_range, p, lw=2)
    plt.title("Predicted Win Probability vs NET Difference")
    plt.xlabel("NET Difference (Team_NET − Opp_NET)")
    plt.ylabel("Predicted P(Win)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[Viz] NET difference curve saved to: {os.path.abspath(out_path)}")


# ---------------------------
# Utility: ensure outputs dir
# ---------------------------
def ensure_dir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------
# Load & Clean
# ---------------------------
def load_data(excel_path: str) -> pd.DataFrame:
    # Loads first sheet by default; adjust sheet_name if needed
    df = pd.read_excel(excel_path)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Coerce numeric columns where expected (errors='ignore' for safety)
    numeric_cols = [
        "Team_Pace", "Opp_Pace", "Team_Seed", "Opp_Seed", "NET_Diff", "Team_NET", "Opp_NET",
        "Team_Turnovers", "Opp_Turnovers", "Team_EFG", "Opp_EFG", "Team_FTA", "Opp_FTA",
        "Team_Rebounds", "Rebounds_Diff", "Team_Won", "Net_OE", "Team_OE", "Opp_OE"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing target
    df = df.dropna(subset=["Team_Won"])

    return df


# ---------------------------
# Feature Engineering
# ---------------------------
REQUESTED_FEATURES = [
    "Team_Pace", "Opp_Pace", "Team_Seed", "Opp_Seed",
    "Team_OE", "Opp_OE"
]

ENGINEERED_DIFFS = {
    "pace_diff": ("Team_Pace", "Opp_Pace"),
    "seed_diff": ("Opp_Seed", "Team_Seed"),  # positive if opponent has worse (higher number) seed
    "net_diff": ("Team_NET", "Opp_NET"),     # positive if team NET rank better (lower is better; diff keeps direction)
    "oe_diff": ("Team_OE", "Opp_OE"),       # positive if team offense is higher
    
}


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    # Base differences
    df["net_diff"] = df["Team_NET"] - df["Opp_NET"]
    df["oe_diff"] = df["Team_OE"] - df["Opp_OE"]
    df["pace_diff"] = df["Team_Pace"] - df["Opp_Pace"]
    df["seed_diff"] = df["Opp_Seed"] - df["Team_Seed"]

    # --- NEW INTERACTION TERMS ---
    df["net_oe_interaction"] = df["net_diff"] * df["oe_diff"]
    df["net_pace_interaction"] = df["net_diff"] * df["pace_diff"]
    df["oe_pace_interaction"] = df["oe_diff"] * df["pace_diff"]
    df["net_seed_interaction"] = df["net_diff"] * df["seed_diff"]

    return df



# ---------------------------
# EDA
# ---------------------------
def eda(df: pd.DataFrame, outdir: str) -> None:
    # Class balance
    target_counts = df["Team_Won"].value_counts().sort_index()
    print("\n[EDA] Team_Won class distribution:")
    print(target_counts)
    print(f"Proportion winning: {target_counts.get(1, 0) / target_counts.sum():.3f}")

    # Correlations (numeric only)
    num_df = df.select_dtypes(include=[np.number]).copy()
    corr = num_df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False)
    plt.title("Correlation Heatmap (numeric features)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "corr_heatmap.png"), dpi=180)
    plt.close()

    # Distributions for requested features
    for col in REQUESTED_FEATURES:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.kdeplot(data=df, x=col, hue="Team_Won", common_norm=False)
            plt.title(f"Distribution by Outcome: {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"dist_{col}.png"), dpi=160)
            plt.close()


# ---------------------------
# Data Split & Preprocess
# ---------------------------
def train_test_prepare(
    df: pd.DataFrame, use_diffs: bool
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:

    # Keep only differences + interactions (drop raw pace/seed/OE to reduce collinearity)
    feature_cols = [
        "net_diff", "oe_diff", "pace_diff", "seed_diff",
        "net_oe_interaction", "net_pace_interaction",
        "oe_pace_interaction", "net_seed_interaction",
    ]

    model_df = df.dropna(subset=feature_cols + ["Team_Won"]).copy()
    X = model_df[feature_cols].copy()
    y = model_df["Team_Won"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n[Split] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, y_train, X_test, y_test, feature_cols


# ---------------------------
# Modeling
# ---------------------------
def build_models(n_jobs: int = -1) -> Dict[str, Pipeline]:
    """
    Returns dictionary of model pipelines keyed by model name.
    Logistic Regression gets scaling; tree-based do not require scaling.
    """
    numeric_features = "passthrough"  # ColumnTransformer kept simple; numeric only

    scaler = StandardScaler(with_mean=True, with_std=True)
    logistic = Pipeline(steps=[
        ("scaler", scaler),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            random_state=RANDOM_STATE,
            max_iter=200
        ))
    ])  


    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE
    )

    gb = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=400,
        max_depth=3,
        random_state=RANDOM_STATE
    )

    return {
        "LogisticRegression": logistic,
        "RandomForest": rf,
        "GradientBoosting": gb
    }


# ---------------------------
# Evaluation Helpers
# ---------------------------
def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    outdir: str,
    feature_names: List[str]
) -> Dict[str, float]:
    print(f"\n===== {name} =====")

    # Fit
    model.fit(X_train, y_train)

    # Predict proba (prob of win = class 1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models without predict_proba
        if hasattr(model, "decision_function"):
            # Map decision function to [0,1] with logistic; approximate
            df_scores = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-df_scores))
        else:
            # As last resort, use predictions as probs (not ideal)
            y_prob = model.predict(X_test)

    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc = np.nan

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"Accuracy: {acc:.3f} | ROC-AUC: {roc:.3f}")
    print("Confusion Matrix [rows=true 0/1, cols=pred 0/1]:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

    # Save ROC curve
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
        plt.title(f"ROC Curve - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"roc_{name}.png"), dpi=160)
        plt.close()
    except Exception as e:
        warnings.warn(f"Could not plot ROC for {name}: {e}")

    # Save Confusion Matrix plot
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"cm_{name}.png"), dpi=160)
        plt.close()
    except Exception as e:
        warnings.warn(f"Could not plot confusion matrix for {name}: {e}")

    # Save predictions CSV for inspection
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_prob"] = y_prob
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(os.path.join(outdir, f"predictions_{name}.csv"), index=False)

    # Model-specific interpretation
    if name == "LogisticRegression":
        interpret_logistic(model, feature_names, outdir)
    else:
        # Tree-based importances & permutation importances
        if hasattr(model, "feature_importances_"):
            ti = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            ti.to_csv(os.path.join(outdir, f"tree_importances_{name}.csv"))
            plt.figure(figsize=(7, 5))
            ti.iloc[:15].plot(kind="bar")
            plt.title(f"Tree Feature Importances - {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"tree_importances_{name}.png"), dpi=160)
            plt.close()

        try:
            perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
            pi = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
            pi.to_csv(os.path.join(outdir, f"perm_importances_{name}.csv"))
        except Exception as e:
            warnings.warn(f"Permutation importance failed for {name}: {e}")

    # SHAP
    if SHAP_AVAILABLE:
        try:
            shap_interpret(name, model, X_train, X_test, feature_names, outdir)
        except Exception as e:
            warnings.warn(f"SHAP interpretation failed for {name}: {e}")

    return {"accuracy": acc, "roc_auc": roc}


def interpret_logistic(pipe_or_model, feature_names: List[str], outdir: str) -> None:
    """Extract coefficients from LogisticRegression inside a pipeline or raw model."""
    # Locate LogisticRegression
    if isinstance(pipe_or_model, Pipeline):
        clf = pipe_or_model.named_steps.get("clf", None)
        scaler = pipe_or_model.named_steps.get("scaler", None)
    else:
        clf = pipe_or_model
        scaler = None

    if not hasattr(clf, "coef_"):
        return
    coefs = clf.coef_.ravel()
    odds_ratios = np.exp(coefs)

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "odds_ratio": odds_ratios
    }).sort_values("odds_ratio", ascending=False)

    coef_df.to_csv(os.path.join(outdir, "logistic_coefficients.csv"), index=False)

    # Bar plot of odds ratios
    plt.figure(figsize=(8, 5))
    sns.barplot(x="odds_ratio", y="feature", data=coef_df, orient="h")
    plt.axvline(1.0, ls="--", c="k")
    plt.title("Logistic Regression Odds Ratios (>1 increases win odds)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "logistic_odds_ratios.png"), dpi=160)
    plt.close()


def shap_interpret(
    name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    outdir: str
) -> None:
    """Run SHAP summary for linear/tree models."""
    # Take a manageable background sample for SHAP speed
    background = shap.sample(X_train, min(1000, len(X_train)), random_state=RANDOM_STATE)

    if isinstance(model, Pipeline):
        base_model = model.named_steps.get("clf", model)
    else:
        base_model = model

    # Try TreeExplainer first, fallback to LinearExplainer/KernelExplainer
    explainer = None
    if hasattr(base_model, "predict_proba") and hasattr(base_model, "feature_importances_"):
        explainer = shap.TreeExplainer(base_model)
    else:
        # Logistic regression path
        try:
            explainer = shap.LinearExplainer(base_model, background)
        except Exception:
            explainer = shap.KernelExplainer(base_model.predict_proba, background)

    # Compute SHAP values for positive class
    if hasattr(base_model, "predict_proba"):
        shap_values = explainer.shap_values(X_test)
        # For tree explainer on binary clf, shap_values can be list [class0, class1]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
    else:
        shap_vals = explainer.shap_values(X_test)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_vals, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"shap_summary_{name}.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_vals, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"shap_bar_{name}.png"), dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------
# Aggregate Insights
# ---------------------------
def summarize_key_insights(
    results: Dict[str, Dict[str, float]],
    df: pd.DataFrame,
    feature_cols: List[str],
    outdir: str
) -> None:
    best_model = max(results, key=lambda k: (np.nan_to_num(results[k]["roc_auc"], nan=-1), results[k]["accuracy"]))
    print(f"\n[Summary] Best model by ROC-AUC then Accuracy: {best_model} "
          f"(AUC={results[best_model]['roc_auc']:.3f}, ACC={results[best_model]['accuracy']:.3f})")

    # Simple directional insight examples (not causal; just conditional associations)
    if {"Team_NET", "Opp_NET", "Team_OE", "Opp_OE"}.issubset(df.columns):
        df_local = df.dropna(subset=["Team_Won", "Team_NET", "Opp_NET", "Team_OE", "Opp_OE"]).copy()
        df_local["better_NET"] = (df_local["Team_NET"] < df_local["Opp_NET"]).astype(int)
        df_local["better_OE"]  = (df_local["Team_OE"]  > df_local["Opp_OE"]).astype(int)

        p_win_better_net = df_local.loc[df_local["better_NET"] == 1, "Team_Won"].mean()
        p_win_worse_net  = df_local.loc[df_local["better_NET"] == 0, "Team_Won"].mean()
        p_win_better_oe  = df_local.loc[df_local["better_OE"]  == 1, "Team_Won"].mean()
        p_win_worse_oe   = df_local.loc[df_local["better_OE"]  == 0, "Team_Won"].mean()

        print(f"\n[Insight] Teams with better Team_NET than Opp_NET: {p_win_better_net:.2%} win rate vs {p_win_worse_net:.2%} when worse.")
        print(f"[Insight] Teams with higher Team_OE than Opp_OE: {p_win_better_oe:.2%} win rate vs {p_win_worse_oe:.2%} when lower.")

        with open(os.path.join(outdir, "insights.txt"), "w") as f:
            f.write(f"Teams with better Team_NET vs Opp_NET have a {p_win_better_net:.1%} win rate (vs {p_win_worse_net:.1%} when worse).\n")
            f.write(f"Teams with higher Team_OE vs Opp_OE have a {p_win_better_oe:.1%} win rate (vs {p_win_worse_oe:.1%} when lower).\n")



# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict March Madness first-round wins.")
    parser.add_argument("--excel", type=str, required=False, help="Path to df_first_round_all_years.xlsx")
    parser.add_argument("--use_diffs", action="store_true",
                        help="Include engineered diff features (pace_diff, seed_diff, net_diff, oe_diff)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallelism for RF/permutation importance")
    args = parser.parse_args()

    outdir = ensure_dir("outputs")

    # Load & clean
    df = load_data(r'C:\Users\888\Desktop\2024 NCAA Stats\df_first_round_all_years.xlsx')
    df = basic_clean(df)
    df = add_diff_features(df)

    # EDA
    eda(df, outdir)

    # Prepare data
    X_train, y_train, X_test, y_test, feature_cols = train_test_prepare(df, use_diffs=args.use_diffs)

    # Build models
    models = build_models(n_jobs=args.n_jobs)

    # Fit & evaluate
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(
            name=name,
            model=model,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            outdir=outdir,
            feature_names=feature_cols
        )
        results[name] = metrics

        # Identify and fit the best model on ALL available rows (for a clean curve)
    best_model_name = max(results, key=lambda k: (np.nan_to_num(results[k]["roc_auc"], nan=-1), results[k]["accuracy"]))
    best_model = build_models(n_jobs=args.n_jobs)[best_model_name]

    # Recreate the exact feature set and training frame used for modeling
    # (reuses your existing function to ensure the same feature_cols)
    _, _, _, _, feature_cols_all = train_test_prepare(df, use_diffs=args.use_diffs)
    model_df_all = df.dropna(subset=feature_cols_all + ["Team_Won"]).copy()
    X_all = model_df_all[feature_cols_all].copy()
    y_all = model_df_all["Team_Won"].astype(int).copy()

    best_model.fit(X_all, y_all)

    # Plot the NET-diff probability curve (only if net_diff exists among features)
    if "net_diff" in feature_cols_all and best_model_name == "LogisticRegression":
        plot_net_diff_curve(best_model, feature_cols_all, out_path=os.path.join(outdir, "net_diff_win_prob_curve.png"))


    coef_csv = os.path.join(outdir, "logistic_coefficients.csv")
    if os.path.exists(coef_csv):
        plot_odds_ratios(coef_csv)


    # Summarize key insights
    summarize_key_insights(results, df, feature_cols, outdir)

    print("\nAll artifacts saved to:", os.path.abspath(outdir))
    print("Done.")


if __name__ == "__main__":
    main()
