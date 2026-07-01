import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple

# Set up matplotlib backend to be non-interactive
import matplotlib
matplotlib.use('Agg')

MODEL_PLOT_DIR = os.path.join("data", "processed", "model_plots")
os.makedirs(MODEL_PLOT_DIR, exist_ok=True)

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes multi-class Brier Score.
    y_true: integer class labels (0, 1, 2)
    y_prob: predicted class probabilities of shape (N, 3)
    """
    y_one_hot = np.zeros_like(y_prob)
    for idx, val in enumerate(y_true):
        y_one_hot[idx, val] = 1.0
    # Formula: 1/N * sum_i(sum_k( (p_ik - y_ik)^2 ))
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))

def plot_calibration(
    y_true: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    class_label: int = 2,
    class_name: str = "Home Win"
) -> str:
    """
    Plots calibration curves for multiple models on a specific outcome class (e.g. Home Win).
    Saves the plot to data/processed/model_plots/calibration_comparison.png.
    """
    plt.figure(figsize=(8, 6))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    
    for model_name, probs in probs_dict.items():
        # Get target class probability
        y_prob_class = probs[:, class_label]
        y_true_binary = (y_true == class_label).astype(float)
        
        # Compute calibration curve
        # Use 5 bins for stability with World Cup dataset sizes
        prob_true, prob_pred = calibration_curve(y_true_binary, y_prob_class, n_bins=5)
        
        plt.plot(prob_pred, prob_true, "s-", label=model_name, linewidth=2)
        
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Actual Win Fraction", fontsize=12)
    plt.title(f"Calibration Curves - {class_name} (Qatar 2022)", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(MODEL_PLOT_DIR, "calibration_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved calibration comparison plot to {plot_path}")
    return plot_path

def evaluate_models(
    df_test: pd.DataFrame,
    poisson_model,
    xgb_model,
    xgb_features: List[str]
) -> Dict:
    """
    Evaluates both models on the test set, computing Log Loss, Brier Score, and generating reports.
    """
    # 1. Prepare target variables
    # Filter out matches where score is missing
    df_clean = df_test.dropna(subset=["home_score", "away_score"]).copy()
    y_true = np.select(
        [df_clean["home_score"] > df_clean["away_score"], df_clean["home_score"] == df_clean["away_score"], df_clean["home_score"] < df_clean["away_score"]],
        [2, 1, 0]
    )
    
    # 2. Get predictions from Poisson Model
    print("Getting predictions from PoissonGoals model...")
    probs_poisson = poisson_model.predict_match_probs(df_clean)
    
    # 3. Get predictions from XGBoost Model
    print("Getting predictions from XGBoostClassifier model...")
    # Prepare classifier features
    df_clean["elo_diff"] = df_clean["home_elo"] - df_clean["away_elo"]
    df_clean["fifa_rank_diff"] = df_clean["home_fifa_rank"] - df_clean["away_fifa_rank"]
    df_clean["fifa_points_diff"] = df_clean["home_fifa_points"] - df_clean["away_fifa_points"]
    df_clean["tm_val_log_ratio"] = np.log1p(df_clean["home_tm_total_value"]) - np.log1p(df_clean["away_tm_total_value"])
    df_clean["tm_val_diff"] = df_clean["home_tm_total_value"] - df_clean["away_tm_total_value"]
    
    X_xgb = df_clean[xgb_features]
    probs_xgb = xgb_model.predict_proba(X_xgb)
    
    # 4. Compute Metrics
    # Poisson
    log_loss_poisson = log_loss(y_true, probs_poisson, labels=[0, 1, 2])
    brier_poisson = compute_brier_score(y_true, probs_poisson)
    
    # XGBoost
    log_loss_xgb = log_loss(y_true, probs_xgb, labels=[0, 1, 2])
    brier_xgb = compute_brier_score(y_true, probs_xgb)
    
    # Print Metrics
    print("\nMODEL PERFORMANCE COMPARISON:")
    print(f"Poisson Goals Model: Log Loss = {log_loss_poisson:.4f}, Brier Score = {brier_poisson:.4f}")
    print(f"XGBoost Classifier:  Log Loss = {log_loss_xgb:.4f}, Brier Score = {brier_xgb:.4f}")
    
    # 5. Save Calibration Plot
    probs_dict = {
        "Poisson Goals Model": probs_poisson,
        "XGBoost Classifier": probs_xgb
    }
    plot_path = plot_calibration(y_true, probs_dict, class_label=2, class_name="Home Win")
    
    # 6. Sample Match Comparison (Qatar 2022 World Cup matches)
    print("\nSAMPLE FIXTURE PREDICTIONS AGREEMENT (Qatar 2022):")
    sample_fixtures = df_clean.tail(5) # Take last 5 matches of the test set
    
    comp_rows = []
    for idx, (f_idx, row) in enumerate(sample_fixtures.iterrows()):
        h_team = row["home_team_standardized"]
        a_team = row["away_team_standardized"]
        h_sc = int(row["home_score"])
        a_sc = int(row["away_score"])
        
        # Predicted probabilities
        # Format: [Away Win, Draw, Home Win]
        p_p = probs_poisson[-(5 - idx)]
        x_p = probs_xgb[-(5 - idx)]
        
        comp_rows.append({
            "Match": f"{h_team} {h_sc}-{a_sc} {a_team}",
            "Poisson Prob (H/D/A)": f"{p_p[2]:.1%}/{p_p[1]:.1%}/{p_p[0]:.1%}",
            "XGBoost Prob (H/D/A)": f"{x_p[2]:.1%}/{x_p[1]:.1%}/{x_p[0]:.1%}"
        })
        
    df_comp = pd.DataFrame(comp_rows)
    print(df_comp.to_string(index=False))
    
    # 7. Print Feature Importance and Significance
    print("\nPOISSON REGRESSION COEFFICIENT SIGNIFICANCE (p-values):")
    summary = poisson_model.model.summary2().tables[1]
    print(summary[["Coef.", "Std.Err.", "z", "P>|z|"]])
    
    print("\nXGBOOST CLASSIFIER FEATURE IMPORTANCE:")
    importances = xgb_model.feature_importances_
    feat_imp = pd.Series(importances, index=xgb_features).sort_values(ascending=False)
    print(feat_imp)
    
    return {
        "poisson": {"log_loss": log_loss_poisson, "brier": brier_poisson},
        "xgb": {"log_loss": log_loss_xgb, "brier": brier_xgb},
        "sample_comp": df_comp.to_dict(orient="records"),
        "calibration_plot_path": plot_path
    }
