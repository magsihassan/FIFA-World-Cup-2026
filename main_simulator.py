import os
import pandas as pd
import numpy as np
from src.models import PoissonMatchModel
from src.simulator import WorldCupSimulator

FEATURED_PATH = os.path.join("data", "processed", "featured_fixtures.csv.gz")

# Canonical 32 teams and group definitions for Qatar 2022
GROUPS_2022 = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"]
}

def extract_team_profiles(df: pd.DataFrame, teams: list, cutoff_date: str) -> dict:
    """
    Extracts the latest pre-tournament feature values for each team prior to the cutoff date.
    Ensures zero lookahead bias.
    """
    profiles = {}
    df_prior = df[df["date"] < cutoff_date].sort_values("date")
    
    for team in teams:
        # Find matches where this team played home or away prior to the tournament
        team_matches = df_prior[
            (df_prior["home_team_standardized"] == team) | 
            (df_prior["away_team_standardized"] == team)
        ]
        
        if len(team_matches) == 0:
            # If no prior match (should not happen), use default/mean profile values
            profiles[team] = {
                "elo": 1500.0,
                "fifa_points": 1000.0,
                "tm_total_value": 0.0,
                "roll_win_rate_10": 0.45,
                "roll_gd_10": 0.0,
                "ewa_win_rate": 0.5,
                "qual_ppg": 1.5,
                "manager_tenure_days": 365.0
            }
            continue
            
        # Take the most recent match
        last_match = team_matches.iloc[-1]
        
        if last_match["home_team_standardized"] == team:
            profiles[team] = {
                "elo": last_match["home_elo"],
                "fifa_points": last_match["home_fifa_points"],
                "tm_total_value": last_match["home_tm_total_value"],
                "roll_win_rate_10": last_match["home_roll_win_rate_10"],
                "roll_gd_10": last_match["home_roll_gd_10"],
                "ewa_win_rate": last_match["home_ewa_win_rate"],
                "qual_ppg": last_match["home_qual_ppg"],
                "manager_tenure_days": last_match["home_manager_tenure_days"]
            }
        else:
            profiles[team] = {
                "elo": last_match["away_elo"],
                "fifa_points": last_match["away_fifa_points"],
                "tm_total_value": last_match["away_tm_total_value"],
                "roll_win_rate_10": last_match["away_roll_win_rate_10"],
                "roll_gd_10": last_match["away_roll_gd_10"],
                "ewa_win_rate": last_match["away_ewa_win_rate"],
                "qual_ppg": last_match["away_qual_ppg"],
                "manager_tenure_days": last_match["away_manager_tenure_days"]
            }
            
    return profiles

def main():
    print("=" * 60)
    print("PHASE 4: WORLD CUP 2022 TOURNAMENT SIMULATOR & VALIDATION")
    print("=" * 60)
    
    # 1. Load featured dataset
    if not os.path.exists(FEATURED_PATH):
        print(f"Featured dataset missing at {FEATURED_PATH}. Regenerating it dynamically...")
        from src.data_loader import load_transfermarkt_data
        from src.feature_builder import build_features
        from main_eda import impute_missing_data, CONSOLIDATED_PATH
        
        tm_data = load_transfermarkt_data()
        df_fixtures = pd.read_csv(CONSOLIDATED_PATH)
        df_fixtures["date"] = pd.to_datetime(df_fixtures["date"])
        
        df_featured = build_features(
            df_fixtures=df_fixtures,
            df_players=tm_data["players"],
            df_valuations=tm_data["player_valuations"],
            df_appearances=tm_data["appearances"],
            df_tm_games=tm_data["games"]
        )
        
        df, df_report = impute_missing_data(df_featured)
        os.makedirs(os.path.dirname(FEATURED_PATH), exist_ok=True)
        df.to_csv(FEATURED_PATH, compression="gzip", index=False)
        print("Featured dataset successfully generated.")
    else:
        print(f"Loading featured dataset from {FEATURED_PATH}...")
        df = pd.read_csv(FEATURED_PATH)
        df["date"] = pd.to_datetime(df["date"])
    
    # 2. Fit Poisson goals model on historical data (before World Cup 2022)
    print("Training Poisson goals model on pre-tournament data...")
    cutoff_date = "2022-11-15"
    df_train = df[df["date"] < cutoff_date].copy()
    
    poisson_model = PoissonMatchModel()
    poisson_model.fit(df_train)
    print("Poisson model successfully trained.")
    print("-" * 40)
    
    # 3. Extract team profiles
    all_teams = []
    for g_teams in GROUPS_2022.values():
        all_teams.extend(g_teams)
        
    print("Extracting pre-tournament team profiles (as of Nov 2022)...")
    team_profiles = extract_team_profiles(df, all_teams, cutoff_date)
    print("Profiles successfully extracted for all 32 teams.")
    print("-" * 40)
    
    # Print sample profiles to verify inputs
    print("SAMPLE PRE-TOURNAMENT PROFILES:")
    sample_teams = ["Argentina", "France", "Croatia", "Morocco", "Qatar"]
    for t in sample_teams:
        p = team_profiles[t]
        print(f" - {t:12} | Elo: {p['elo']:.0f} | FIFA Pts: {p['fifa_points']:.0f} | TM Value: {p['tm_total_value']/1e6:.1f}M EUR | Roll GD: {p['roll_gd_10']:.2f}")
    print("-" * 40)
    
    # 4. Run simulations
    num_sims = 10000
    print(f"Running {num_sims:,} full tournament simulations...")
    
    simulator = WorldCupSimulator(poisson_model, GROUPS_2022)
    
    # Sim trackers: team -> count
    stats = {t: {"r16": 0, "quarters": 0, "semis": 0, "final": 0, "champion": 0} for t in all_teams}
    
    for sim_idx in range(num_sims):
        res = simulator.simulate_tournament(team_profiles)
        
        # Increment counts
        stats[res["champion"]]["champion"] += 1
        stats[res["runner_up"]]["final"] += 1
        stats[res["champion"]]["final"] += 1 # Champion also reached final
        
        # Semis
        for team in res["semis"]:
            stats[team]["semis"] += 1
            stats[team]["final"] = stats[team].get("final", 0) # ensure init
            
        # Quarters
        for team in res["quarters"]:
            stats[team]["quarters"] += 1
            
        # R16
        for team in res["r16"]:
            stats[team]["r16"] += 1
            
    # Calculate probabilities
    probs_list = []
    for team, counts in stats.items():
        probs_list.append({
            "Team": team,
            "R16 %": (counts["r16"] / num_sims) * 100,
            "QF %": (counts["quarters"] / num_sims) * 100,
            "SF %": (counts["semis"] / num_sims) * 100,
            "Final %": (counts["final"] / num_sims) * 100,
            "Champ %": (counts["champion"] / num_sims) * 100
        })
        
    df_probs = pd.DataFrame(probs_list).sort_values(by="Champ %", ascending=False)
    
    # Display top 15 teams
    print("\nTOP 15 TEAMS BY PREDICTED CHAMPION PROBABILITY:")
    print(df_probs.head(15).to_string(index=False, formatters={
        "R16 %": "{:.1f}%".format,
        "QF %": "{:.1f}%".format,
        "SF %": "{:.1f}%".format,
        "Final %": "{:.1f}%".format,
        "Champ %": "{:.1f}%".format
    }))
    print("-" * 40)
    
    # 5. Validation Check
    print("\nVALIDATION: QATAR 2022 ACTUAL TOP 4 OUTCOMES VS SIMULATED PROBABILITIES")
    print("=" * 60)
    actual_top4 = [
        {"Team": "Argentina", "Actual Finish": "Champion"},
        {"Team": "France", "Actual Finish": "Runner-Up"},
        {"Team": "Croatia", "Actual Finish": "3rd Place"},
        {"Team": "Morocco", "Actual Finish": "4th Place"}
    ]
    
    validation_rows = []
    for item in actual_top4:
        team = item["Team"]
        row_probs = df_probs[df_probs["Team"] == team].iloc[0]
        validation_rows.append({
            "Team": team,
            "Actual Finish": item["Actual Finish"],
            "QF Rank": int(df_probs.sort_values("QF %", ascending=False)["Team"].tolist().index(team) + 1),
            "SF Rank": int(df_probs.sort_values("SF %", ascending=False)["Team"].tolist().index(team) + 1),
            "Champ Rank": int(df_probs.sort_values("Champ %", ascending=False)["Team"].tolist().index(team) + 1),
            "Champ Prob": f"{row_probs['Champ %']:.2f}%",
            "SF Prob": f"{row_probs['SF %']:.2f}%",
            "R16 Prob": f"{row_probs['R16 %']:.2f}%"
        })
        
    df_val = pd.DataFrame(validation_rows)
    print(df_val.to_string(index=False))
    print("=" * 60)
    
    print("\nSIMULATION COMPLETE!")

if __name__ == "__main__":
    main()
