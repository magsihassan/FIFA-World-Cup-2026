import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from typing import Dict

# Set up matplotlib backend to be non-interactive
import matplotlib
matplotlib.use('Agg')

EDA_PLOT_DIR = os.path.join("data", "processed", "eda_plots")
os.makedirs(EDA_PLOT_DIR, exist_ok=True)

def run_eda_analysis(df_featured: pd.DataFrame) -> Dict:
    """
    Performs Exploratory Data Analysis:
    - Calculates correlations of features against match outcome
    - Saves sanity check plots (Elo diff vs win rate, squad value vs win rate)
    - Saves distribution plots for key features
    Returns a dictionary of correlation results.
    """
    print("Running Exploratory Data Analysis...")
    df = df_featured.copy()
    
    # 1. Define target variables
    # Filter out matches where score is missing (unplayed or invalid)
    df = df.dropna(subset=["home_score", "away_score"])
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["home_won"] = (df["goal_diff"] > 0).astype(float)
    df["match_result"] = np.select(
        [df["goal_diff"] > 0, df["goal_diff"] == 0, df["goal_diff"] < 0],
        [1.0, 0.5, 0.0]
    )
    
    # Calculate differences for strength/squad values
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["fifa_rank_diff"] = df["home_fifa_rank"] - df["away_fifa_rank"]
    df["fifa_points_diff"] = df["home_fifa_points"] - df["away_fifa_points"]
    
    # Transfermarkt value log ratio (to handle skewness)
    df["tm_val_log_ratio"] = np.log1p(df["home_tm_total_value"]) - np.log1p(df["away_tm_total_value"])
    df["tm_val_diff"] = df["home_tm_total_value"] - df["away_tm_total_value"]
    
    # Select key features for correlation matrix
    correlation_features = [
        "goal_diff", "home_won", "match_result",
        "elo_diff", "fifa_points_diff", "fifa_rank_diff",
        "tm_val_diff", "tm_val_log_ratio",
        "home_roll_win_rate_10", "away_roll_win_rate_10",
        "home_roll_gd_10", "away_roll_gd_10",
        "home_ewa_win_rate", "away_ewa_win_rate",
        "h2h_home_win_rate", "h2h_home_gd",
        "home_manager_tenure_days", "away_manager_tenure_days",
        "home_qual_ppg", "home_qual_gd",
        "host_nation_boost"
    ]
    
    # Filter to existing columns in df
    avail_features = [c for c in correlation_features if c in df.columns]
    
    # Compute Pearson Correlation
    corr_matrix = df[avail_features].corr(method="pearson")
    
    # Print top correlations with match_result
    corr_target = corr_matrix["match_result"].sort_values(ascending=False)
    print("\nTop Correlations with Match Result (Home Win=1.0, Draw=0.5, Away Win=0.0):")
    print(corr_target)
    
    # 2. Plot Correlation Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Matrix of Engineered Features", fontsize=14)
    plt.tight_layout()
    corr_heatmap_path = os.path.join(EDA_PLOT_DIR, "correlation_heatmap.png")
    plt.savefig(corr_heatmap_path, dpi=150)
    plt.close()
    print(f"Saved correlation heatmap to {corr_heatmap_path}")
    
    # 3. Sanity Check Plot 1: Elo Difference vs Win Rate
    # Bin Elo difference in 100-point increments
    df_elo_valid = df.dropna(subset=["elo_diff", "home_won"])
    if len(df_elo_valid) > 100:
        df_elo_valid["elo_bin"] = (df_elo_valid["elo_diff"] // 100) * 100
        # Group by bin and compute win rate
        elo_grouped = df_elo_valid.groupby("elo_bin")["home_won"].agg(["mean", "count"]).reset_index()
        # Filter bins with at least 15 matches for statistical stability
        elo_grouped = elo_grouped[elo_grouped["count"] >= 15]
        
        plt.figure(figsize=(8, 5))
        plt.plot(elo_grouped["elo_bin"], elo_grouped["mean"], "o-", color="crimson", linewidth=2)
        plt.axhline(0.5, linestyle="--", color="gray")
        plt.axvline(0, linestyle="--", color="gray")
        plt.xlabel("Elo Difference (Home - Away)")
        plt.ylabel("Home Win Rate")
        plt.title("Elo Difference vs. Actual Home Win Rate")
        plt.grid(True, alpha=0.3)
        elo_plot_path = os.path.join(EDA_PLOT_DIR, "elo_diff_vs_winrate.png")
        plt.savefig(elo_plot_path, dpi=150)
        plt.close()
        print(f"Saved Elo check plot to {elo_plot_path}")
        
    # 4. Sanity Check Plot 2: Transfermarkt Log Value Ratio vs Win Rate
    df_tm_valid = df.dropna(subset=["tm_val_log_ratio", "home_won"])
    # Only keep matches that actually have Transfermarkt data (where tm_has_data == 1.0)
    df_tm_valid = df_tm_valid[df_tm_valid["home_tm_has_data"] == 1.0]
    if len(df_tm_valid) > 100:
        # Bin log ratio in 0.5 increments
        df_tm_valid["tm_bin"] = (df_tm_valid["tm_val_log_ratio"] // 0.5) * 0.5
        tm_grouped = df_tm_valid.groupby("tm_bin")["home_won"].agg(["mean", "count"]).reset_index()
        tm_grouped = tm_grouped[tm_grouped["count"] >= 10]
        
        plt.figure(figsize=(8, 5))
        plt.plot(tm_grouped["tm_bin"], tm_grouped["mean"], "s-", color="navy", linewidth=2)
        plt.axhline(0.5, linestyle="--", color="gray")
        plt.axvline(0, linestyle="--", color="gray")
        plt.xlabel("Log Squad Value Ratio (Home Value / Away Value)")
        plt.ylabel("Home Win Rate")
        plt.title("Transfermarkt Log Value Ratio vs. Actual Home Win Rate")
        plt.grid(True, alpha=0.3)
        tm_plot_path = os.path.join(EDA_PLOT_DIR, "tm_value_vs_winrate.png")
        plt.savefig(tm_plot_path, dpi=150)
        plt.close()
        print(f"Saved Transfermarkt check plot to {tm_plot_path}")
        
    # 5. Distribution Checks for Key Features
    plt.figure(figsize=(12, 4))
    
    # ELO Diff
    plt.subplot(1, 3, 1)
    sns.histplot(df["elo_diff"].dropna(), kde=True, color="crimson", bins=30)
    plt.title("Elo Difference Distribution")
    plt.xlabel("Home Elo - Away Elo")
    
    # TM Log Value Ratio
    plt.subplot(1, 3, 2)
    sns.histplot(df[df["home_tm_has_data"] == 1.0]["tm_val_log_ratio"].dropna(), kde=True, color="navy", bins=30)
    plt.title("TM Log Value Ratio Distribution")
    plt.xlabel("Log Squad Value Ratio")
    
    # 10-match Rolling GD
    plt.subplot(1, 3, 3)
    sns.histplot(df["home_roll_gd_10"].dropna(), kde=True, color="teal", bins=30)
    plt.title("10-Match Rolling GD (Home)")
    plt.xlabel("Average Goal Diff")
    
    plt.tight_layout()
    dist_plot_path = os.path.join(EDA_PLOT_DIR, "feature_distributions.png")
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"Saved distributions plot to {dist_plot_path}")
    
    # Package results to return
    results = {
        "correlation_target": corr_target.to_dict(),
        "paths": {
            "heatmap": corr_heatmap_path,
            "elo_plot": elo_plot_path if 'elo_plot_path' in locals() else None,
            "tm_plot": tm_plot_path if 'tm_plot_path' in locals() else None,
            "dist_plot": dist_plot_path
        }
    }
    return results
