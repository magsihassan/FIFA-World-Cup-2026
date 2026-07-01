import os
import pandas as pd
import numpy as np
from typing import Tuple
from src.data_loader import (
    load_transfermarkt_data,
    load_international_results,
    load_fifa_world_cup,
    load_fifa_rankings
)
from src.elo_scraper import download_teams_list, build_elo_history_df
from src.feature_builder import build_features
from src.eda import run_eda_analysis

PROCESSED_DIR = os.path.join("data", "processed")
CONSOLIDATED_PATH = os.path.join(PROCESSED_DIR, "consolidated_fixtures.csv.gz")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "featured_fixtures.csv.gz")

def impute_missing_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Explicitly documents and applies imputation strategies for missing values.
    Returns:
    - df_imputed: The dataframe with applied imputations
    - df_report: A report detailing missing counts and strategies
    """
    print("Applying explicit imputation strategies for missing values...")
    df_imputed = df.copy()
    
    # Track missing statistics before imputation
    missing_stats = []
    
    # 1. Elo Ratings
    # If missing, impute with 1500 (standard default) and set indicator flag
    for prefix in ["home_", "away_"]:
        col = f"{prefix}elo"
        missing_count = df_imputed[col].isna().sum()
        pct = missing_count / len(df_imputed)
        missing_stats.append({
            "feature": col,
            "missing_count": missing_count,
            "percentage": f"{pct:.1%}",
            "strategy": "Impute with 1500, add binary flag"
        })
        df_imputed[f"{col}_imputed"] = df_imputed[col].isna().astype(float)
        df_imputed[col] = df_imputed[col].fillna(1500.0)
        # Elo Rank
        df_imputed[f"{col}_rank"] = df_imputed[f"{col}_rank"].fillna(100.0) # default to mid-rank
        
    # 2. FIFA Rankings
    # Missing for matches before 1993. Impute with 0.0 points / rank 200.0 and set flag.
    for prefix in ["home_", "away_"]:
        col_pts = f"{prefix}fifa_points"
        col_rk = f"{prefix}fifa_rank"
        
        missing_count = df_imputed[col_pts].isna().sum()
        pct = missing_count / len(df_imputed)
        missing_stats.append({
            "feature": col_pts,
            "missing_count": missing_count,
            "percentage": f"{pct:.1%}",
            "strategy": "Impute with 0.0 (expected pre-1993), add binary flag"
        })
        df_imputed[f"{prefix}fifa_imputed"] = df_imputed[col_pts].isna().astype(float)
        df_imputed[col_pts] = df_imputed[col_pts].fillna(0.0)
        df_imputed[col_rk] = df_imputed[col_rk].fillna(200.0) # low rank default
        
    # 3. Transfermarkt Squad Quality
    # Missing for older eras (pre-2004) or smaller countries.
    # Set tm_has_data flag to 0.0, impute total value to 0, age/depth to overall medians.
    median_age = df_imputed["home_tm_avg_age"].median()
    if pd.isna(median_age):
        median_age = 26.0
    median_depth = df_imputed["home_tm_depth_dropoff"].median()
    if pd.isna(median_depth):
        median_depth = 2.5
        
    for prefix in ["home_", "away_"]:
        col_val = f"{prefix}tm_total_value"
        col_age = f"{prefix}tm_avg_age"
        col_depth = f"{prefix}tm_depth_dropoff"
        col_caps = f"{prefix}tm_avg_caps"
        col_form = f"{prefix}tm_top_players_form"
        col_has = f"{prefix}tm_has_data"
        
        missing_count = df_imputed[col_val].isna().sum()
        pct = missing_count / len(df_imputed)
        missing_stats.append({
            "feature": col_val,
            "missing_count": missing_count,
            "percentage": f"{pct:.1%}",
            "strategy": "Fill with 0, set tm_has_data to 0.0"
        })
        
        # Apply fills
        df_imputed[col_has] = df_imputed[col_has].fillna(0.0)
        df_imputed[col_val] = df_imputed[col_val].fillna(0.0)
        df_imputed[f"{prefix}tm_avg_value"] = df_imputed[f"{prefix}tm_avg_value"].fillna(0.0)
        df_imputed[col_age] = df_imputed[col_age].fillna(median_age)
        df_imputed[col_depth] = df_imputed[col_depth].fillna(median_depth)
        df_imputed[col_caps] = df_imputed[col_caps].fillna(0.0)
        df_imputed[col_form] = df_imputed[col_form].fillna(0.0)
        
    # 4. Manager Tenure
    # Impute missing tenure with median tenure (e.g. 1 year / 365 days) and set flag
    median_tenure = df_imputed["home_manager_tenure_days"].median()
    if pd.isna(median_tenure):
        median_tenure = 365.0
        
    for prefix in ["home_", "away_"]:
        col_ten = f"{prefix}manager_tenure_days"
        missing_count = df_imputed[col_ten].isna().sum()
        pct = missing_count / len(df_imputed)
        missing_stats.append({
            "feature": col_ten,
            "missing_count": missing_count,
            "percentage": f"{pct:.1%}",
            "strategy": f"Impute with median ({median_tenure} days), add flag"
        })
        df_imputed[f"{col_ten}_imputed"] = df_imputed[col_ten].isna().astype(float)
        df_imputed[col_ten] = df_imputed[col_ten].fillna(median_tenure)
        
    # 5. Qualification Stats
    # Impute missing with 0 and set flag (usually means team did not participate or data missing)
    for prefix in ["home_", "away_"]:
        col_ppg = f"{prefix}qual_ppg"
        col_gd = f"{prefix}qual_gd"
        col_playoff = f"{prefix}qual_played_playoff"
        
        missing_count = df_imputed[col_ppg].isna().sum()
        pct = missing_count / len(df_imputed)
        missing_stats.append({
            "feature": col_ppg,
            "missing_count": missing_count,
            "percentage": f"{pct:.1%}",
            "strategy": "Impute with 0.0, add flag"
        })
        df_imputed[f"{prefix}qual_imputed"] = df_imputed[col_ppg].isna().astype(float)
        df_imputed[col_ppg] = df_imputed[col_ppg].fillna(0.0)
        df_imputed[col_gd] = df_imputed[col_gd].fillna(0.0)
        df_imputed[col_playoff] = df_imputed[col_playoff].fillna(0.0)
        
    df_report = pd.DataFrame(missing_stats)
    return df_imputed, df_report

def main():
    print("=" * 60)
    print("PHASE 2: FEATURE ENGINEERING & EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # 1. Load player-level Transfermarkt tables
    print("1. Loading detailed Transfermarkt player/appearance tables...")
    tm_data = load_transfermarkt_data()
    df_players = tm_data["players"]
    df_valuations = tm_data["player_valuations"]
    df_appearances = tm_data["appearances"]
    df_games = tm_data["games"]
    print("Transfermarkt tables successfully loaded.")
    print("-" * 40)
    
    # 2. Load consolidated fixtures
    print("2. Loading consolidated fixtures from Phase 1...")
    if not os.path.exists(CONSOLIDATED_PATH):
        # Fallback to recreate if missing
        print("Consolidated fixtures file missing. Running Phase 1 loaders to recreate...")
        df_results = load_international_results()["results"]
        df_shootouts = load_international_results()["shootouts"]
        df_wc_matches = load_fifa_world_cup()["matches"]
        df_rankings = load_fifa_rankings()
        teams_list = download_teams_list()
        df_elo = build_elo_history_df()
        df_tm_nt = tm_data["national_teams"]
        from src.data_merger import merge_datasets
        df_fixtures, _ = merge_datasets(df_results, df_shootouts, df_wc_matches, df_rankings, df_elo, df_tm_nt, teams_list)
    else:
        df_fixtures = pd.read_csv(CONSOLIDATED_PATH)
        df_fixtures["date"] = pd.to_datetime(df_fixtures["date"])
    print(f"Loaded {len(df_fixtures)} fixtures.")
    print("-" * 40)
    
    # 3. Build features
    print("3. Building engineered features...")
    df_featured = build_features(
        df_fixtures=df_fixtures,
        df_players=df_players,
        df_valuations=df_valuations,
        df_appearances=df_appearances,
        df_tm_games=df_games
    )
    print(f"Features created! Total columns: {len(df_featured.columns)}")
    print("-" * 40)
    
    # 4. Impute missing values
    df_imputed, df_report = impute_missing_data(df_featured)
    
    print("\nIMPUTATION REPORT:")
    print(df_report.to_string(index=False))
    print("-" * 40)
    
    # 5. Save final featured dataframe
    print(f"5. Saving final featured dataset to {OUTPUT_PATH}...")
    df_imputed.to_csv(OUTPUT_PATH, compression="gzip", index=False)
    print("Featured dataset saved successfully.")
    print("-" * 40)
    
    # 6. Run EDA
    print("6. Executing Exploratory Data Analysis & Plotting...")
    eda_results = run_eda_analysis(df_imputed)
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
