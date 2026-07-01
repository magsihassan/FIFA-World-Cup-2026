import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from src.name_standardizer import standardize_team_name

PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def merge_datasets(
    df_results: pd.DataFrame,
    df_shootouts: pd.DataFrame,
    df_wc_matches: pd.DataFrame,
    df_rankings: pd.DataFrame,
    df_elo: pd.DataFrame,
    df_tm_national_teams: pd.DataFrame,
    teams_list_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    Merges all datasets into a single fixture-level DataFrame.
    Returns:
        - merged_df: The consolidated DataFrame
        - stats: A dictionary containing merge statistics and sample mismatches
    """
    stats = {}
    
    # 1. Prepare Base Fixtures (martj42's international results)
    df_fixtures = df_results.copy()
    df_fixtures["date"] = pd.to_datetime(df_fixtures["date"])
    
    # Track row counts
    stats["initial_fixtures_count"] = len(df_fixtures)
    
    # Standardize names in base fixtures
    df_fixtures["home_team_clean"] = df_fixtures["home_team"].apply(standardize_team_name)
    df_fixtures["away_team_clean"] = df_fixtures["away_team"].apply(standardize_team_name)
    
    # Create code lookup maps for ELO ratings
    code_to_name = dict(zip(teams_list_df["code"], teams_list_df["name"]))
    # ELO codes mapped to standardized names
    standardized_elo_names = {code: standardize_team_name(name) for code, name in code_to_name.items()}
    name_to_elo_code = {v: k for k, v in standardized_elo_names.items()}
    
    # Map fixtures home/away to ELO team codes
    df_fixtures["home_elo_code"] = df_fixtures["home_team_clean"].map(name_to_elo_code)
    df_fixtures["away_elo_code"] = df_fixtures["away_team_clean"].map(name_to_elo_code)
    
    # ELO Mismatch Stats
    home_elo_mismatch = df_fixtures[df_fixtures["home_elo_code"].isna()]["home_team_clean"].unique().tolist()
    away_elo_mismatch = df_fixtures[df_fixtures["away_elo_code"].isna()]["away_team_clean"].unique().tolist()
    all_elo_mismatches = sorted(list(set(home_elo_mismatch + away_elo_mismatch)))
    stats["elo_name_mismatches_count"] = len(all_elo_mismatches)
    stats["elo_name_mismatches_sample"] = all_elo_mismatches[:10]
    
    # 2. Match ELO Ratings (Time-series join)
    # Sort for merge_asof
    df_fixtures = df_fixtures.sort_values("date")
    
    # Prepare ELO history
    df_elo_clean = df_elo.copy()
    df_elo_clean["date"] = pd.to_datetime(df_elo_clean["date"])
    df_elo_clean = df_elo_clean.sort_values("date")
    
    # Join home team ELO
    df_fixtures = pd.merge_asof(
        df_fixtures,
        df_elo_clean[["date", "team_code", "rating", "rank"]].rename(columns={"rating": "home_elo", "rank": "home_elo_rank"}),
        on="date",
        left_by="home_elo_code",
        right_by="team_code",
        direction="backward",
        allow_exact_matches=False # strictly before match date
    )
    # Join away team ELO
    df_fixtures = pd.merge_asof(
        df_fixtures,
        df_elo_clean[["date", "team_code", "rating", "rank"]].rename(columns={"rating": "away_elo", "rank": "away_elo_rank"}),
        on="date",
        left_by="away_elo_code",
        right_by="team_code",
        direction="backward",
        allow_exact_matches=False # strictly before match date
    )
    
    # 3. Match FIFA Rankings (monthly rankings)
    df_rankings_clean = df_rankings.copy()
    df_rankings_clean["rank_date"] = pd.to_datetime(df_rankings_clean["rank_date"])
    df_rankings_clean["country_clean"] = df_rankings_clean["country_full"].apply(standardize_team_name)
    df_rankings_clean = df_rankings_clean.sort_values("rank_date")
    
    # Track FIFA Ranking name mismatches
    fixture_teams = set(df_fixtures["home_team_clean"].unique().tolist() + df_fixtures["away_team_clean"].unique().tolist())
    ranking_teams = set(df_rankings_clean["country_clean"].unique().tolist())
    fifa_mismatches = sorted(list(fixture_teams - ranking_teams))
    stats["fifa_ranking_name_mismatches_count"] = len(fifa_mismatches)
    stats["fifa_ranking_name_mismatches_sample"] = fifa_mismatches[:10]
    
    # We will use merge_asof to match FIFA rank on/before the match date
    # Let's rename rank_date to date for merging
    df_rankings_clean = df_rankings_clean.rename(columns={"rank_date": "date"})
    
    # Join home team FIFA rank
    df_fixtures = pd.merge_asof(
        df_fixtures,
        df_rankings_clean[["date", "country_clean", "rank", "total_points"]].rename(columns={"rank": "home_fifa_rank", "total_points": "home_fifa_points"}),
        on="date",
        left_by="home_team_clean",
        right_by="country_clean",
        direction="backward"
    )
    df_fixtures = df_fixtures.drop(columns=["country_clean"], errors="ignore")
    
    # Join away team FIFA rank
    df_fixtures = pd.merge_asof(
        df_fixtures,
        df_rankings_clean[["date", "country_clean", "rank", "total_points"]].rename(columns={"rank": "away_fifa_rank", "total_points": "away_fifa_points"}),
        on="date",
        left_by="away_team_clean",
        right_by="country_clean",
        direction="backward"
    )
    df_fixtures = df_fixtures.drop(columns=["country_clean"], errors="ignore")
    
    # 4. Match Transfermarkt Squad Stats
    df_tm = df_tm_national_teams.copy()
    df_tm["country_clean"] = df_tm["country_name"].apply(standardize_team_name)
    
    # Track Transfermarkt name mismatches (natural because it only tracks 124 teams)
    tm_teams = set(df_tm["country_clean"].unique().tolist())
    tm_mismatches = sorted(list(fixture_teams - tm_teams))
    stats["transfermarkt_name_mismatches_count"] = len(tm_mismatches)
    stats["transfermarkt_name_mismatches_sample"] = tm_mismatches[:10]
    
    # Since Transfermarkt is a static snapshot (last season's data), we perform a standard merge
    # Join home squad stats
    df_fixtures = df_fixtures.merge(
        df_tm[["country_clean", "squad_size", "average_age", "total_market_value", "coach_name"]].rename(
            columns={
                "squad_size": "home_tm_squad_size",
                "average_age": "home_tm_avg_age",
                "total_market_value": "home_tm_market_value",
                "coach_name": "home_tm_coach"
            }
        ),
        left_on="home_team_clean",
        right_on="country_clean",
        how="left"
    ).drop(columns=["country_clean"], errors="ignore")
    
    # Join away squad stats
    df_fixtures = df_fixtures.merge(
        df_tm[["country_clean", "squad_size", "average_age", "total_market_value", "coach_name"]].rename(
            columns={
                "squad_size": "away_tm_squad_size",
                "average_age": "away_tm_avg_age",
                "total_market_value": "away_tm_market_value",
                "coach_name": "away_tm_coach"
            }
        ),
        left_on="away_team_clean",
        right_on="country_clean",
        how="left"
    ).drop(columns=["country_clean"], errors="ignore")
    
    # 5. Shootouts integration (add shootout winner flag if match ended in shootout)
    df_shootouts_clean = df_shootouts.copy()
    df_shootouts_clean["date"] = pd.to_datetime(df_shootouts_clean["date"])
    df_shootouts_clean["home_team_clean"] = df_shootouts_clean["home_team"].apply(standardize_team_name)
    df_shootouts_clean["away_team_clean"] = df_shootouts_clean["away_team"].apply(standardize_team_name)
    
    df_fixtures = df_fixtures.merge(
        df_shootouts_clean[["date", "home_team_clean", "away_team_clean", "winner"]].rename(columns={"winner": "shootout_winner"}),
        on=["date", "home_team_clean", "away_team_clean"],
        how="left"
    )
    
    # 6. Add World Cup Match details from WorldCupMatches
    # We clean WorldCupMatches
    df_wc = df_wc_matches.copy()
    # Handle dates like '13 Jul 1930 - 15:00'
    df_wc["Datetime"] = pd.to_datetime(df_wc["Datetime"], errors="coerce")
    df_wc["date"] = df_wc["Datetime"].dt.normalize()
    df_wc["home_team_clean"] = df_wc["Home Team Name"].apply(standardize_team_name)
    df_wc["away_team_clean"] = df_wc["Away Team Name"].apply(standardize_team_name)
    df_wc = df_wc.dropna(subset=["date"])
    
    df_fixtures = df_fixtures.merge(
        df_wc[["date", "home_team_clean", "away_team_clean", "Stage", "Attendance", "Referee"]].rename(
            columns={
                "Stage": "wc_stage",
                "Attendance": "wc_attendance",
                "Referee": "wc_referee"
            }
        ).drop_duplicates(subset=["date", "home_team_clean", "away_team_clean"]),
        on=["date", "home_team_clean", "away_team_clean"],
        how="left"
    )
    
    # 7. Add Placeholders for Next Stage Features
    placeholder_cols = [
        "home_last_5_goals_avg", "away_last_5_goals_avg",
        "home_last_5_conceded_avg", "away_last_5_conceded_avg",
        "home_rest_days", "away_rest_days",
        "h2h_home_wins", "h2h_away_wins", "h2h_draws"
    ]
    for col in placeholder_cols:
        df_fixtures[col] = np.nan
        
    # Drop intermediate columns
    df_fixtures = df_fixtures.drop(columns=["home_elo_code", "away_elo_code", "team_code_x", "team_code_y"], errors="ignore")
    
    # Rename columns to standard ones
    df_fixtures = df_fixtures.rename(columns={
        "home_team_clean": "home_team_standardized",
        "away_team_clean": "away_team_standardized"
    })
    
    # Final counts
    stats["final_fixtures_count"] = len(df_fixtures)
    stats["matches_with_elo_both"] = len(df_fixtures[df_fixtures["home_elo"].notna() & df_fixtures["away_elo"].notna()])
    stats["matches_with_fifa_both"] = len(df_fixtures[df_fixtures["home_fifa_points"].notna() & df_fixtures["away_fifa_points"].notna()])
    stats["matches_with_tm_both"] = len(df_fixtures[df_fixtures["home_tm_market_value"].notna() & df_fixtures["away_tm_market_value"].notna()])
    
    # Save the consolidated DataFrame
    output_path = os.path.join(PROCESSED_DIR, "consolidated_fixtures.csv.gz")
    df_fixtures.to_csv(output_path, compression="gzip", index=False)
    stats["saved_path"] = output_path
    
    return df_fixtures, stats
