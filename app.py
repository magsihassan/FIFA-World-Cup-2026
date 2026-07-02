import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from src.models import PoissonMatchModel, get_classifier_features, train_xgb_classifier
from src.simulator import WorldCupSimulator
from main_simulator import GROUPS_2022, extract_team_profiles

app = Flask(__name__)
CORS(app)

import time
from collections import defaultdict

# Simple In-Memory Rate Limiter (no external package dependencies)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 100  # requests per window
ip_request_history = defaultdict(list)

@app.before_request
def check_rate_limit():
    # Only rate limit endpoints under /api/
    if request.path.startswith("/api/"):
        ip = request.remote_addr or "127.0.0.1"
        now = time.time()
        # Filter history to keep only requests in the active window
        ip_request_history[ip] = [t for t in ip_request_history[ip] if now - t < RATE_LIMIT_WINDOW]
        
        if len(ip_request_history[ip]) >= RATE_LIMIT_MAX_REQUESTS:
            return jsonify({"error": "Too many requests. Rate limit exceeded (max 100 req/min)."}), 429
            
        ip_request_history[ip].append(now)

FEATURED_PATH = os.path.join("data", "processed", "featured_fixtures.csv.gz")

# Global variables to store trained models and team profiles
poisson_model = None
xgb_model = None
xgb_features = None
team_profiles = None
df_featured = None

def init_models():
    global poisson_model, xgb_model, xgb_features, team_profiles, df_featured
    print("Initializing models and extracting profiles...")
    
    # 1. Regenerate features if missing, or load directly
    if not os.path.exists(FEATURED_PATH):
        print("Featured dataset missing. Running dynamic generation...")
        from src.data_loader import load_transfermarkt_data
        from src.feature_builder import build_features
        from main_eda import impute_missing_data, CONSOLIDATED_PATH
        
        tm_data = load_transfermarkt_data()
        df_fixtures = pd.read_csv(CONSOLIDATED_PATH)
        df_fixtures["date"] = pd.to_datetime(df_fixtures["date"])
        
        df_feat = build_features(
            df_fixtures=df_fixtures,
            df_players=tm_data["players"],
            df_valuations=tm_data["player_valuations"],
            df_appearances=tm_data["appearances"],
            df_tm_games=tm_data["games"]
        )
        df_featured, _ = impute_missing_data(df_feat)
        os.makedirs(os.path.dirname(FEATURED_PATH), exist_ok=True)
        df_featured.to_csv(FEATURED_PATH, compression="gzip", index=False)
    else:
        df_featured = pd.read_csv(FEATURED_PATH)
        df_featured["date"] = pd.to_datetime(df_featured["date"])
        
    # 2. Fit Poisson goals model on pre-tournament data
    cutoff_date = "2022-11-15"
    df_train = df_featured[df_featured["date"] < cutoff_date].copy()
    
    poisson_model = PoissonMatchModel()
    poisson_model.fit(df_train)
    
    # 3. Train XGBoost multiclass classifier
    xgb_model, xgb_features = train_xgb_classifier(df_train)
    
    # 4. Extract team profiles
    all_teams = []
    for g_teams in GROUPS_2022.values():
        all_teams.extend(g_teams)
    team_profiles = extract_team_profiles(df_featured, all_teams, cutoff_date)
    print("Models and profiles initialized successfully!")

# Initialize on import
init_models()

@app.route("/api/teams", methods=["GET"])
def get_teams():
    """Returns the list of 32 teams and their profiles."""
    return jsonify({
        "groups": GROUPS_2022,
        "profiles": team_profiles
    })

@app.route("/api/predict", methods=["GET"])
def predict_match():
    """
    Predicts scoreline probabilities and comparison stats between two teams.
    """
    home_team = request.args.get("home_team")
    away_team = request.args.get("away_team")
    
    if not home_team or not away_team:
        return jsonify({"error": "Missing home_team or away_team parameter"}), 400
        
    if home_team not in team_profiles or away_team not in team_profiles:
        return jsonify({"error": "Invalid team name"}), 400
        
    if home_team == away_team:
        return jsonify({"error": "Home and away teams must be different"}), 400
        
    profile_h = team_profiles[home_team]
    profile_a = team_profiles[away_team]
    
    # Predict expected goals (lambda)
    host_boost_h = 1.0 if home_team == "Qatar" else 0.0
    host_boost_a = 1.0 if away_team == "Qatar" else 0.0
    
    # Use helper predictor
    # We construct a mock DataFrame row to reuse predict_match_probs
    df_mock = pd.DataFrame([{
        "home_team_standardized": home_team,
        "away_team_standardized": away_team,
        "country": "Qatar",
        "home_elo": profile_h["elo"],
        "away_elo": profile_a["elo"],
        "home_fifa_points": profile_h["fifa_points"],
        "away_fifa_points": profile_a["fifa_points"],
        "home_tm_total_value": profile_h["tm_total_value"],
        "away_tm_total_value": profile_a["tm_total_value"],
        "home_roll_win_rate_10": profile_h["roll_win_rate_10"],
        "away_roll_win_rate_10": profile_a["roll_win_rate_10"],
        "home_roll_gd_10": profile_h["roll_gd_10"],
        "away_roll_gd_10": profile_a["roll_gd_10"],
        "home_ewa_win_rate": profile_h["ewa_win_rate"],
        "away_ewa_win_rate": profile_a["ewa_win_rate"],
        "home_qual_ppg": profile_h["qual_ppg"],
        "away_qual_ppg": profile_a["qual_ppg"],
        "home_manager_tenure_days": profile_h["manager_tenure_days"],
        "away_manager_tenure_days": profile_a["manager_tenure_days"],
        "is_home": 0.0,
        "host_nation_boost": host_boost_h,
        "home_score": np.nan,
        "away_score": np.nan
    }])
    
    probs = poisson_model.predict_match_probs(df_mock)[0]
    p_away_win, p_draw, p_home_win = probs[0], probs[1], probs[2]
    
    # Expected goals
    lambda_h = poisson_model.model.predict([1.0, 
        profile_h["elo"] - profile_a["elo"],
        profile_h["fifa_points"] - profile_a["fifa_points"],
        np.log1p(profile_h["tm_total_value"]) - np.log1p(profile_a["tm_total_value"]),
        profile_h["roll_win_rate_10"] - profile_a["roll_win_rate_10"],
        profile_h["roll_gd_10"] - profile_a["roll_gd_10"],
        profile_h["ewa_win_rate"] - profile_a["ewa_win_rate"],
        0.0, # is_home
        host_boost_h,
        profile_h["qual_ppg"] - profile_a["qual_ppg"],
        profile_h["manager_tenure_days"] - profile_a["manager_tenure_days"]
    ])[0]
    
    lambda_a = poisson_model.model.predict([1.0, 
        profile_a["elo"] - profile_h["elo"],
        profile_a["fifa_points"] - profile_h["fifa_points"],
        np.log1p(profile_a["tm_total_value"]) - np.log1p(profile_h["tm_total_value"]),
        profile_a["roll_win_rate_10"] - profile_h["roll_win_rate_10"],
        profile_a["roll_gd_10"] - profile_h["roll_gd_10"],
        profile_a["ewa_win_rate"] - profile_h["ewa_win_rate"],
        0.0,
        host_boost_a,
        profile_a["qual_ppg"] - profile_h["qual_ppg"],
        profile_a["manager_tenure_days"] - profile_h["manager_tenure_days"]
    ])[0]
    
    # Scoreline heatmap (0-3 score grids)
    from scipy.stats import poisson as pois_dist
    heatmap = {}
    for h_score in range(4):
        for a_score in range(4):
            prob = pois_dist.pmf(h_score, lambda_h) * pois_dist.pmf(a_score, lambda_a)
            heatmap[f"{h_score}-{a_score}"] = round(float(prob) * 100, 1)
            
    # Add a catch-all 3+ category
    h_3plus = 1.0 - sum(pois_dist.pmf(i, lambda_h) for i in range(3))
    a_3plus = 1.0 - sum(pois_dist.pmf(i, lambda_a) for i in range(3))
    heatmap["3+-3+"] = round(float(h_3plus * a_3plus) * 100, 1)
    
    return jsonify({
        "probs": {
            "home": round(p_home_win * 100, 1),
            "draw": round(p_draw * 100, 1),
            "away": round(p_away_win * 100, 1)
        },
        "lambdas": {
            "home": round(lambda_h, 2),
            "away": round(lambda_a, 2)
        },
        "heatmap": heatmap,
        "comparison": {
            "elo": {"home": int(profile_h["elo"]), "away": int(profile_a["elo"])},
            "fifa": {"home": int(profile_h["fifa_points"]), "away": int(profile_a["fifa_points"])},
            "market_value": {"home": profile_h["tm_total_value"], "away": profile_a["tm_total_value"]},
            "form": {"home": round(profile_h["roll_win_rate_10"] * 100), "away": round(profile_a["roll_win_rate_10"] * 100)}
        }
    })

@app.route("/api/simulate", methods=["POST"])
def run_simulation():
    """
    Runs Monte Carlo simulations and returns probabilities and a sample bracket.
    """
    try:
        req_data = request.json or {}
        runs_val = req_data.get("runs", 1000)
        num_sims = int(runs_val)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid 'runs' parameter. Must be an integer."}), 400
        
    if num_sims <= 0:
        return jsonify({"error": "'runs' must be a positive integer."}), 400
        
    # Cap at 2000 in UI to maintain quick API response
    num_sims = min(num_sims, 2000)
    
    simulator = WorldCupSimulator(poisson_model, GROUPS_2022)
    stats = {t: {"r16": 0, "quarters": 0, "semis": 0, "final": 0, "champion": 0} for t in team_profiles.keys()}
    
    sample_bracket = None
    for idx in range(num_sims):
        res = simulator.simulate_tournament(team_profiles)
        
        # Save the first run as our visual sample bracket
        if idx == 0:
            sample_bracket = res
            
        stats[res["champion"]]["champion"] += 1
        stats[res["runner_up"]]["final"] += 1
        stats[res["champion"]]["final"] += 1
        
        for team in res["semis"]:
            stats[team]["semis"] += 1
        for team in res["quarters"]:
            stats[team]["quarters"] += 1
        for team in res["r16"]:
            stats[team]["r16"] += 1
            
    probs_list = []
    for team, counts in stats.items():
        probs_list.append({
            "team": team,
            "r16": round((counts["r16"] / num_sims) * 100, 1),
            "quarters": round((counts["quarters"] / num_sims) * 100, 1),
            "semis": round((counts["semis"] / num_sims) * 100, 1),
            "final": round((counts["final"] / num_sims) * 100, 1),
            "champion": round((counts["champion"] / num_sims) * 100, 1)
        })
        
    probs_sorted = sorted(probs_list, key=lambda x: x["champion"], reverse=True)
    
    return jsonify({
        "probs": probs_sorted,
        "sample_bracket": sample_bracket
    })

@app.route("/api/insights", methods=["GET"])
def get_insights():
    """Returns coefficient significances and classifier feature importances."""
    # Coefficients from Poisson Model
    summary_df = poisson_model.model.summary2().tables[1]
    coefs = []
    for feat, row in summary_df.iterrows():
        coefs.append({
            "feature": feat,
            "coef": round(row["Coef."], 4),
            "p_value": round(row["P>|z|"], 4),
            "significant": bool(row["P>|z|"] < 0.05)
        })
        
    # XGBoost Classifier Importances
    importances = xgb_model.feature_importances_
    features_list = get_classifier_features()
    imp_list = []
    for feat, imp in zip(features_list, importances):
        imp_list.append({
            "feature": feat,
            "importance": round(float(imp) * 100, 2)
        })
    imp_sorted = sorted(imp_list, key=lambda x: x["importance"], reverse=True)
    
    return jsonify({
        "poisson_coefs": coefs,
        "xgb_importances": imp_sorted,
        "metrics": {
            "poisson_logloss": 0.9995,
            "poisson_brier": 0.5484,
            "xgb_logloss": 1.0505,
            "xgb_brier": 0.5756
        }
    })

@app.route("/")
def serve_spa():
    """Serves the Single Page Application index file."""
    return render_template("index.html")
