import os
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Tuple, List, Optional
from src.name_standardizer import standardize_team_name

# Configuration for half-life decay (EWA)
HALF_LIFE_DAYS = 1095.0  # 3 years
LAMBDA_DECAY = np.log(2.0) / HALF_LIFE_DAYS

def compute_rolling_features(df_fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling match metrics for each team (Win/Draw/Loss rate, Goal Difference)
    over the last 5, 10, and 20 matches.
    """
    print("Computing rolling features...")
    df = df_fixtures.copy()
    
    # We need to construct a long dataframe where each row is a team-match observation
    # to easily compute rolling metrics.
    match_rows = []
    for idx, row in df.iterrows():
        # Home team perspective
        match_rows.append({
            "match_idx": idx,
            "date": row["date"],
            "team": row["home_team_standardized"],
            "opponent": row["away_team_standardized"],
            "goals_for": row["home_score"],
            "goals_against": row["away_score"],
            "is_home": 1,
            "result": 1.0 if row["home_score"] > row["away_score"] else (0.5 if row["home_score"] == row["away_score"] else 0.0)
        })
        # Away team perspective
        match_rows.append({
            "match_idx": idx,
            "date": row["date"],
            "team": row["away_team_standardized"],
            "opponent": row["home_team_standardized"],
            "goals_for": row["away_score"],
            "goals_against": row["home_score"],
            "is_home": 0,
            "result": 1.0 if row["away_score"] > row["home_score"] else (0.5 if row["away_score"] == row["home_score"] else 0.0)
        })
        
    df_teams = pd.DataFrame(match_rows)
    df_teams = df_teams.sort_values(by=["team", "date"])
    
    # Compute rolling values
    df_teams["goals_diff"] = df_teams["goals_for"] - df_teams["goals_against"]
    
    # Placeholders for results
    for window in [5, 10, 20]:
        # Shift by 1 to make it strictly BEFORE the match
        df_teams[f"roll_win_rate_{window}"] = df_teams.groupby("team")["result"].shift(1).rolling(window, min_periods=1).mean()
        df_teams[f"roll_gd_{window}"] = df_teams.groupby("team")["goals_diff"].shift(1).rolling(window, min_periods=1).mean()
        
    # Re-merge back into the fixture-level dataframe
    df_home = df_teams[df_teams["is_home"] == 1].set_index("match_idx")
    df_away = df_teams[df_teams["is_home"] == 0].set_index("match_idx")
    
    for window in [5, 10, 20]:
        df[f"home_roll_win_rate_{window}"] = df_home[f"roll_win_rate_{window}"]
        df[f"home_roll_gd_{window}"] = df_home[f"roll_gd_{window}"]
        df[f"away_roll_win_rate_{window}"] = df_away[f"roll_win_rate_{window}"]
        df[f"away_roll_gd_{window}"] = df_away[f"roll_gd_{window}"]
        
    return df

def compute_ewa_features(df_fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Computes exponentially time-weighted win rate and goal difference over full history.
    """
    print("Computing exponentially time-weighted features...")
    df = df_fixtures.copy()
    
    # We will build a running EWA for each team
    # S_n = S_{n-1} * exp(-lambda * dt) + x_n * (1 - exp(-lambda * dt))
    # Let's track the running EWA state for each team
    team_states = {} # team -> (last_date, ewa_win_rate, ewa_gd)
    
    home_ewa_win = []
    home_ewa_gd = []
    away_ewa_win = []
    away_ewa_gd = []
    
    for idx, row in df.iterrows():
        date = row["date"]
        h_team = row["home_team_standardized"]
        a_team = row["away_team_standardized"]
        h_score = row["home_score"]
        a_score = row["away_score"]
        
        # Calculate current match results
        if pd.isna(h_score) or pd.isna(a_score):
            h_res, a_res = 0.5, 0.5
            h_gd, a_gd = 0.0, 0.0
        else:
            h_res = 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)
            a_res = 1.0 - h_res
            h_gd = h_score - a_score
            a_gd = -h_gd
            
        # Get prior EWA for Home
        if h_team in team_states:
            last_date, ewa_win, ewa_gd = team_states[h_team]
            dt = (date - last_date).days
            factor = np.exp(-LAMBDA_DECAY * dt)
            # Record EWA before match
            home_ewa_win.append(ewa_win)
            home_ewa_gd.append(ewa_gd)
            # Update state with this match
            new_ewa_win = ewa_win * factor + h_res * (1.0 - factor)
            new_ewa_gd = ewa_gd * factor + h_gd * (1.0 - factor)
            team_states[h_team] = (date, new_ewa_win, new_ewa_gd)
        else:
            # First match ever: default to 0.5 win rate, 0 gd
            home_ewa_win.append(0.5)
            home_ewa_gd.append(0.0)
            team_states[h_team] = (date, h_res, h_gd)
            
        # Get prior EWA for Away
        if a_team in team_states:
            last_date, ewa_win, ewa_gd = team_states[a_team]
            dt = (date - last_date).days
            factor = np.exp(-LAMBDA_DECAY * dt)
            # Record EWA before match
            away_ewa_win.append(ewa_win)
            away_ewa_gd.append(ewa_gd)
            # Update state with this match
            new_ewa_win = ewa_win * factor + a_res * (1.0 - factor)
            new_ewa_gd = ewa_gd * factor + a_gd * (1.0 - factor)
            team_states[a_team] = (date, new_ewa_win, new_ewa_gd)
        else:
            away_ewa_win.append(0.5)
            away_ewa_gd.append(0.0)
            team_states[a_team] = (date, a_res, a_gd)
            
    df["home_ewa_win_rate"] = home_ewa_win
    df["home_ewa_gd"] = home_ewa_gd
    df["away_ewa_win_rate"] = away_ewa_win
    df["away_ewa_gd"] = away_ewa_gd
    
    return df

def compute_h2h_features(df_fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Computes historical head-to-head records between the two specific teams.
    """
    print("Computing head-to-head features...")
    df = df_fixtures.copy()
    
    # Track head-to-head history: frozenset({teamA, teamB}) -> list of matches
    h2h_history = {}
    
    h2h_win_rate = []
    h2h_gd = []
    h2h_last_result = [] # 1 for home win, 0 for draw, -1 for loss, NaN for none
    
    for idx, row in df.iterrows():
        h_team = row["home_team_standardized"]
        a_team = row["away_team_standardized"]
        h_score = row["home_score"]
        a_score = row["away_score"]
        
        pair = frozenset({h_team, a_team})
        
        if pair in h2h_history:
            prior_matches = h2h_history[pair]
            
            # Compute stats from prior matches
            wins = 0
            draws = 0
            gd_sum = 0.0
            last_res = np.nan
            
            for p_match in prior_matches:
                p_home = p_match["home"]
                p_h_score = p_match["h_score"]
                p_a_score = p_match["a_score"]
                
                # Check outcome from home team's perspective
                if p_home == h_team:
                    if p_h_score > p_a_score:
                        wins += 1
                        last_res = 1.0
                    elif p_h_score == p_a_score:
                        draws += 1
                        last_res = 0.0
                    else:
                        last_res = -1.0
                    gd_sum += (p_h_score - p_a_score)
                else:
                    # p_home is away team
                    if p_a_score > p_h_score:
                        wins += 1
                        last_res = 1.0
                    elif p_a_score == p_h_score:
                        draws += 1
                        last_res = 0.0
                    else:
                        last_res = -1.0
                    gd_sum += (p_a_score - p_h_score)
                    
            total_prior = len(prior_matches)
            h2h_win_rate.append(wins / total_prior)
            h2h_gd.append(gd_sum / total_prior)
            h2h_last_result.append(last_res)
            
            # Append this match to history
            prior_matches.append({"home": h_team, "h_score": h_score, "a_score": a_score})
        else:
            h2h_win_rate.append(0.5) # no prior history, default to neutral
            h2h_gd.append(0.0)
            h2h_last_result.append(np.nan)
            
            h2h_history[pair] = [{"home": h_team, "h_score": h_score, "a_score": a_score}]
            
    df["h2h_home_win_rate"] = h2h_win_rate
    df["h2h_home_gd"] = h2h_gd
    df["h2h_last_result"] = h2h_last_result
    
    return df

def build_transfermarkt_squad_features(
    df_fixtures: pd.DataFrame,
    df_players: pd.DataFrame,
    df_valuations: pd.DataFrame,
    df_appearances: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes Transfermarkt squad quality features (total value, avg value, avg age, bench dropoff, caps, player forms)
    for home and away teams.
    """
    print("Computing Transfermarkt squad-level features...")
    df = df_fixtures.copy()
    
    # 1. Standardize player citizenships
    df_players["citizenship_clean"] = df_players["country_of_citizenship"].apply(standardize_team_name)
    df_players["date_of_birth"] = pd.to_datetime(df_players["date_of_birth"], errors="coerce")
    
    # 2. Set up valuations mapping
    df_valuations["date"] = pd.to_datetime(df_valuations["date"])
    df_valuations = df_valuations.sort_values("date")
    
    # 3. Pre-aggregate player values per year to speed up joins
    # For each player, we can find their valuation at the end of each year
    df_valuations["year"] = df_valuations["date"].dt.year
    df_player_yearly_val = df_valuations.groupby(["player_id", "year"])["market_value_in_eur"].last().reset_index()
    player_val_dict = df_player_yearly_val.set_index(["player_id", "year"])["market_value_in_eur"].to_dict()
    
    # 4. Pre-calculate caps for each player per year from appearances
    df_appearances["date"] = pd.to_datetime(df_appearances["date"])
    df_appearances["year"] = df_appearances["date"].dt.year
    # Total appearances for a player before or during a year (cumulative caps)
    df_player_caps = df_appearances.groupby(["player_id", "year"]).size().groupby(level=0).cumsum().reset_index(name="caps")
    player_caps_dict = df_player_caps.set_index(["player_id", "year"])["caps"].to_dict()
    
    # 5. Pre-calculate player form (goals+assists in last 10 appearances)
    # To keep this fast, let's calculate average goals+assists per appearance for each player in each year
    df_appearances["goals_assists"] = df_appearances["goals"].fillna(0) + df_appearances["assists"].fillna(0)
    df_player_form = df_appearances.groupby(["player_id", "year"])["goals_assists"].mean().reset_index(name="form")
    player_form_dict = df_player_form.set_index(["player_id", "year"])["form"].to_dict()
    
    # We will build a dictionary: (country, year) -> squad_features_dict
    squad_cache = {}
    
    # Group players by citizenship
    players_by_country = {country: grp for country, grp in df_players.groupby("citizenship_clean")}
    
    def get_squad_metrics(country: str, year: int) -> Dict:
        key = (country, year)
        if key in squad_cache:
            return squad_cache[key]
            
        if country not in players_by_country:
            squad_cache[key] = {}
            return {}
            
        country_players = players_by_country[country]
        
        # Determine player value and age for the year vectorially
        dob_years = country_players["date_of_birth"].dt.year
        age = year - dob_years
        valid_mask = (age >= 17) & (age <= 38)
        df_active = country_players[valid_mask].copy()
        
        if len(df_active) == 0:
            squad_cache[key] = {}
            return {}
            
        # Get values, caps, and form vectorially
        p_ids = df_active["player_id"].values
        df_active["value"] = [player_val_dict.get((pid, year), np.nan) for pid in p_ids]
        df_active["value"] = df_active["value"].fillna(df_active["market_value_in_eur"]).fillna(50000.0)
        
        df_active["caps"] = [player_caps_dict.get((pid, year), 0) for pid in p_ids]
        df_active["form"] = [player_form_dict.get((pid, year), 0.0) for pid in p_ids]
        df_active["age"] = year - df_active["date_of_birth"].dt.year
        
        # Sort by value to find the top 23 (squad)
        df_active = df_active.sort_values(by="value", ascending=False).head(23)
        
        total_value = df_active["value"].sum()
        avg_value = df_active["value"].mean()
        avg_age = df_active["age"].mean()
        avg_caps = df_active["caps"].mean()
        
        # Squad depth (top 11 vs next 12)
        top_11_val = df_active["value"].head(11).sum()
        next_12_val = df_active["value"].tail(12).sum()
        depth_dropoff = top_11_val / max(next_12_val, 1.0)
        
        # Recent form of top 3 players
        top_3_form = df_active.head(3)["form"].mean()
        
        metrics = {
            "tm_total_value": total_value,
            "tm_avg_value": avg_value,
            "tm_avg_age": avg_age,
            "tm_avg_caps": avg_caps,
            "tm_depth_dropoff": depth_dropoff,
            "tm_top_players_form": top_3_form,
            "tm_has_data": 1.0
        }
        squad_cache[key] = metrics
        return metrics

    # Map features to fixtures
    print("Mapping squad metrics to matches...")
    home_vals = []
    away_vals = []
    
    for idx, row in df.iterrows():
        year = row["date"].year
        h_team = row["home_team_standardized"]
        a_team = row["away_team_standardized"]
        
        # Get home metrics
        h_metrics = get_squad_metrics(h_team, year)
        # Get away metrics
        a_metrics = get_squad_metrics(a_team, year)
        
        home_vals.append(h_metrics)
        away_vals.append(a_metrics)
        
    df_home_tm = pd.DataFrame(home_vals)
    df_away_tm = pd.DataFrame(away_vals)
    
    # Add columns with suffixes
    for col in df_home_tm.columns:
        df[f"home_{col}"] = df_home_tm[col]
    for col in df_away_tm.columns:
        df[f"away_{col}"] = df_away_tm[col]
        
    # Injuries and suspensions placeholder (curated per tournament)
    df["home_injuries_flag"] = 0.0
    df["away_injuries_flag"] = 0.0
    
    return df

def compute_manager_features(df_fixtures: pd.DataFrame, df_tm_games: pd.DataFrame) -> pd.DataFrame:
    """
    Computes manager tenure, win percentage, and major-tournament track records.
    """
    print("Computing manager features...")
    df = df_fixtures.copy()
    
    # 1. Clean games dataframe
    df_games_clean = df_tm_games.dropna(subset=["home_club_manager_name", "away_club_manager_name"]).copy()
    df_games_clean["date"] = pd.to_datetime(df_games_clean["date"])
    df_games_clean["home_club_name_clean"] = df_games_clean["home_club_name"].apply(standardize_team_name)
    df_games_clean["away_club_name_clean"] = df_games_clean["away_club_name"].apply(standardize_team_name)
    
    # Build manager mapping over time
    # We will build a list of manager changes for each team
    # Format: team -> list of (start_date, manager_name)
    manager_history = {}
    
    # We process games chronologically to construct manager history
    df_games_clean = df_games_clean.sort_values("date")
    for _, row in df_games_clean.iterrows():
        date = row["date"]
        h_team = row["home_club_name_clean"]
        a_team = row["away_club_name_clean"]
        h_manager = row["home_club_manager_name"]
        a_manager = row["away_club_manager_name"]
        
        # Update Home Team Manager
        if h_team not in manager_history:
            manager_history[h_team] = [(date, h_manager)]
        else:
            last_mgr = manager_history[h_team][-1][1]
            if last_mgr != h_manager:
                manager_history[h_team].append((date, h_manager))
                
        # Update Away Team Manager
        if a_team not in manager_history:
            manager_history[a_team] = [(date, a_manager)]
        else:
            last_mgr = manager_history[a_team][-1][1]
            if last_mgr != a_manager:
                manager_history[a_team].append((date, a_manager))
                
    # Function to get manager and their start date at match time
    def get_manager_at_time(team: str, date: pd.Timestamp) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
        if team not in manager_history:
            return None, None
        history = manager_history[team]
        current_mgr = None
        start_date = None
        for mgr_date, mgr_name in history:
            if mgr_date <= date:
                current_mgr = mgr_name
                start_date = mgr_date
            else:
                break
        return current_mgr, start_date

    home_tenure = []
    away_tenure = []
    
    for idx, row in df.iterrows():
        date = row["date"]
        h_team = row["home_team_standardized"]
        a_team = row["away_team_standardized"]
        
        # Home Manager
        h_mgr, h_start = get_manager_at_time(h_team, date)
        if h_mgr and h_start:
            home_tenure.append((date - h_start).days)
        else:
            home_tenure.append(np.nan)
            
        # Away Manager
        a_mgr, a_start = get_manager_at_time(a_team, date)
        if a_mgr and a_start:
            away_tenure.append((date - a_start).days)
        else:
            away_tenure.append(np.nan)
            
    df["home_manager_tenure_days"] = home_tenure
    df["away_manager_tenure_days"] = away_tenure
    
    # Manager win percentage placeholder (since it requires historical games mapping,
    # we default to a standard 0.45 win percentage if missing)
    df["home_manager_win_pct"] = np.nan
    df["away_manager_win_pct"] = np.nan
    
    return df

def compute_qualification_features(df_fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Computes qualification performance metrics (points-per-game, goal difference, playoff flag)
    for the current tournament cycle (Y-2 to Y).
    """
    print("Computing qualification features...")
    df = df_fixtures.copy()
    
    # Filter to only qualification matches
    df_qual = df[df["tournament"].str.contains("qualification|qualifying", case=False, na=False)].copy()
    
    # 1. Melt df_qual to get team-level qualification matches
    rows = []
    for prefix in ["home_", "away_"]:
        opp_prefix = "away_" if prefix == "home_" else "home_"
        df_p = df_qual[[
            "date", 
            f"{prefix}team_standardized", 
            f"{opp_prefix}team_standardized",
            f"{prefix}score", 
            f"{opp_prefix}score",
            "tournament"
        ]].copy()
        df_p.columns = ["date", "team", "opponent", "score_for", "score_against", "tournament"]
        df_p["is_playoff"] = df_p["tournament"].str.contains("play-off|playoff", case=False, na=False).astype(float)
        rows.append(df_p)
    df_qual_melt = pd.concat(rows, ignore_index=True).dropna(subset=["score_for", "score_against"])
    df_qual_melt["year"] = df_qual_melt["date"].dt.year
    df_qual_melt["gd"] = df_qual_melt["score_for"] - df_qual_melt["score_against"]
    df_qual_melt["points"] = np.select(
        [df_qual_melt["score_for"] > df_qual_melt["score_against"], df_qual_melt["score_for"] == df_qual_melt["score_against"]],
        [3.0, 1.0],
        default=0.0
    )
    
    # Group by team
    qual_by_team = {team: grp for team, grp in df_qual_melt.groupby("team")}
    
    qual_cache = {}
    
    def get_qual_stats(team: str, year: int) -> Dict:
        key = (team, year)
        if key in qual_cache:
            return qual_cache[key]
            
        if team not in qual_by_team:
            qual_cache[key] = {}
            return {}
            
        grp = qual_by_team[team]
        # Filter qualification matches in the 3-year window [year-2, year] vectorially
        cycle_matches = grp[(grp["year"] >= year - 2) & (grp["year"] <= year)]
        
        if len(cycle_matches) == 0:
            qual_cache[key] = {}
            return {}
            
        total = len(cycle_matches)
        stats = {
            "qual_ppg": cycle_matches["points"].sum() / total,
            "qual_gd": cycle_matches["gd"].sum() / total,
            "qual_win_rate": (cycle_matches["points"] == 3.0).sum() / total,
            "qual_played_playoff": float(cycle_matches["is_playoff"].max() > 0)
        }
        qual_cache[key] = stats
        return stats

    home_ppg, home_gd, home_playoff = [], [], []
    away_ppg, away_gd, away_playoff = [], [], []
    
    for idx, row in df.iterrows():
        year = row["date"].year
        h_team = row["home_team_standardized"]
        a_team = row["away_team_standardized"]
        
        h_stats = get_qual_stats(h_team, year)
        a_stats = get_qual_stats(a_team, year)
        
        home_ppg.append(h_stats.get("qual_ppg", np.nan))
        home_gd.append(h_stats.get("qual_gd", np.nan))
        home_playoff.append(h_stats.get("qual_played_playoff", np.nan))
        
        away_ppg.append(a_stats.get("qual_ppg", np.nan))
        away_gd.append(a_stats.get("qual_gd", np.nan))
        away_playoff.append(a_stats.get("qual_played_playoff", np.nan))
        
    df["home_qual_ppg"] = home_ppg
    df["home_qual_gd"] = home_gd
    df["home_qual_played_playoff"] = home_playoff
    df["away_qual_ppg"] = away_ppg
    df["away_qual_gd"] = away_gd
    df["away_qual_played_playoff"] = away_playoff
    
    return df

def compute_tournament_context(df_fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Computes tournament context features:
    - host_nation_boost flag
    - Group draw difficulty (placeholder for predictions)
    - Tournament experience (placeholder for previous squad stats)
    """
    print("Computing tournament context features...")
    df = df_fixtures.copy()
    
    # 1. Home/Away/Neutral logic
    # is_home: 1 if home playing at home (not neutral), else 0
    df["is_home"] = df["neutral"].apply(lambda x: 0.0 if x else 1.0)
    df["is_away"] = df["neutral"].apply(lambda x: 0.0 if x else 0.0) # away never has home advantage
    
    # 2. host_nation_boost
    # 1.0 if World Cup match and home_team is playing in their own country
    host_boost = []
    for idx, row in df.iterrows():
        is_wc = "world cup" in row["tournament"].lower() and "qualification" not in row["tournament"].lower()
        is_host = row["home_team_standardized"] == row["country"] or row["away_team_standardized"] == row["country"]
        if is_wc and is_host:
            host_boost.append(1.0)
        else:
            host_boost.append(0.0)
    df["host_nation_boost"] = host_boost
    
    # 3. Group draw difficulty and tournament experience placeholders
    df["home_group_difficulty"] = np.nan
    df["away_group_difficulty"] = np.nan
    df["home_tournament_experience"] = np.nan
    df["away_tournament_experience"] = np.nan
    
    return df

def build_features(
    df_fixtures: pd.DataFrame,
    df_players: pd.DataFrame,
    df_valuations: pd.DataFrame,
    df_appearances: pd.DataFrame,
    df_tm_games: pd.DataFrame
) -> pd.DataFrame:
    """Orchestrates the entire feature building process."""
    df = df_fixtures.copy()
    df = compute_rolling_features(df)
    df = compute_ewa_features(df)
    df = compute_h2h_features(df)
    df = build_transfermarkt_squad_features(df, df_players, df_valuations, df_appearances)
    df = compute_manager_features(df, df_tm_games)
    df = compute_qualification_features(df)
    df = compute_tournament_context(df)
    
    # Clean up any duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    return df
