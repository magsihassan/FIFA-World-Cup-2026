import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson
from xgboost import XGBClassifier
from typing import Tuple, Dict, List

def prepare_poisson_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Stacks match data into team-level rows to train a unified Poisson regression goals model.
    Each match (Team A vs Team B) yields two rows:
    - Row 1: Goals scored by Team A, using Team A's relative features vs Team B
    - Row 2: Goals scored by Team B, using Team B's relative features vs Team A
    """
    stacked_rows = []
    
    for idx, row in df.iterrows():
        # Home team perspective
        h_val = row["home_tm_total_value"]
        a_val = row["away_tm_total_value"]
        tm_val_log_ratio = np.log1p(h_val) - np.log1p(a_val)
        
        stacked_rows.append({
            "goals": row["home_score"],
            "elo_diff": row["home_elo"] - row["away_elo"],
            "fifa_points_diff": row["home_fifa_points"] - row["away_fifa_points"],
            "tm_val_log_ratio": tm_val_log_ratio,
            "roll_win_rate_diff": row["home_roll_win_rate_10"] - row["away_roll_win_rate_10"],
            "roll_gd_diff": row["home_roll_gd_10"] - row["away_roll_gd_10"],
            "ewa_win_rate_diff": row["home_ewa_win_rate"] - row["away_ewa_win_rate"],
            "is_home": row["is_home"],
            "host_boost": row["host_nation_boost"],
            "qual_ppg_diff": row["home_qual_ppg"] - row["away_qual_ppg"],
            "manager_tenure_diff": row["home_manager_tenure_days"] - row["away_manager_tenure_days"]
        })
        
        # Away team perspective
        stacked_rows.append({
            "goals": row["away_score"],
            "elo_diff": row["away_elo"] - row["home_elo"],
            "fifa_points_diff": row["away_fifa_points"] - row["home_fifa_points"],
            "tm_val_log_ratio": -tm_val_log_ratio,
            "roll_win_rate_diff": row["away_roll_win_rate_10"] - row["home_roll_win_rate_10"],
            "roll_gd_diff": row["away_roll_gd_10"] - row["home_roll_gd_10"],
            "ewa_win_rate_diff": row["away_ewa_win_rate"] - row["home_ewa_win_rate"],
            "is_home": 0.0, # Away team is not home (even on neutral, it gets 0.0)
            "host_boost": row["host_nation_boost"] if row["away_team_standardized"] == row["country"] else 0.0,
            "qual_ppg_diff": row["away_qual_ppg"] - row["home_qual_ppg"],
            "manager_tenure_diff": row["away_manager_tenure_days"] - row["home_manager_tenure_days"]
        })
        
    df_stacked = pd.DataFrame(stacked_rows).dropna()
    y = df_stacked["goals"]
    X = df_stacked.drop(columns=["goals"])
    X = sm.add_constant(X)
    return y, X

class PoissonMatchModel:
    """
    Goal prediction model using Poisson regression.
    Predicts expected goals scored and solves for match outcome probabilities.
    """
    def __init__(self):
        self.model = None
        self.features = None
        
    def fit(self, df_train: pd.DataFrame):
        y, X = prepare_poisson_data(df_train)
        self.features = X.columns.tolist()
        # Fit GLM with Poisson family and Log link
        self.model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        return self
        
    def predict_match_probs(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Predicts match outcome probabilities (Away Win, Draw, Home Win) for test set.
        Returns:
            - A 2D array of shape (N, 3) where columns correspond to classes [0, 1, 2]
        """
        probs = []
        max_goals = 15
        
        for idx, row in df_test.iterrows():
            # Home team features relative to away team
            h_val = row["home_tm_total_value"]
            a_val = row["away_tm_total_value"]
            tm_val_log_ratio = np.log1p(h_val) - np.log1p(a_val)
            
            # Predict expected home goals (lambda_H)
            h_feat = {
                "const": 1.0,
                "elo_diff": row["home_elo"] - row["away_elo"],
                "fifa_points_diff": row["home_fifa_points"] - row["away_fifa_points"],
                "tm_val_log_ratio": tm_val_log_ratio,
                "roll_win_rate_diff": row["home_roll_win_rate_10"] - row["away_roll_win_rate_10"],
                "roll_gd_diff": row["home_roll_gd_10"] - row["away_roll_gd_10"],
                "ewa_win_rate_diff": row["home_ewa_win_rate"] - row["away_ewa_win_rate"],
                "is_home": row["is_home"],
                "host_boost": row["host_nation_boost"],
                "qual_ppg_diff": row["home_qual_ppg"] - row["away_qual_ppg"],
                "manager_tenure_diff": row["home_manager_tenure_days"] - row["away_manager_tenure_days"]
            }
            # Make sure keys align with self.features
            h_vector = [h_feat[col] for col in self.features if col != "const"]
            h_vector_with_const = [1.0] + h_vector
            lambda_H = self.model.predict(h_vector_with_const)[0]
            
            # Predict expected away goals (lambda_A)
            a_feat = {
                "const": 1.0,
                "elo_diff": row["away_elo"] - row["home_elo"],
                "fifa_points_diff": row["away_fifa_points"] - row["home_fifa_points"],
                "tm_val_log_ratio": -tm_val_log_ratio,
                "roll_win_rate_diff": row["away_roll_win_rate_10"] - row["home_roll_win_rate_10"],
                "roll_gd_diff": row["away_roll_gd_10"] - row["home_roll_gd_10"],
                "ewa_win_rate_diff": row["away_ewa_win_rate"] - row["home_ewa_win_rate"],
                "is_home": 0.0,
                "host_boost": row["host_nation_boost"] if row["away_team_standardized"] == row["country"] else 0.0,
                "qual_ppg_diff": row["away_qual_ppg"] - row["home_qual_ppg"],
                "manager_tenure_diff": row["away_manager_tenure_days"] - row["home_manager_tenure_days"]
            }
            a_vector = [a_feat[col] for col in self.features if col != "const"]
            a_vector_with_const = [1.0] + a_vector
            lambda_A = self.model.predict(a_vector_with_const)[0]
            
            # Calculate score probability grid
            goals_h_pdf = poisson.pmf(np.arange(max_goals), lambda_H)
            goals_a_pdf = poisson.pmf(np.arange(max_goals), lambda_A)
            
            # Outer product to get score grid
            score_grid = np.outer(goals_h_pdf, goals_a_pdf)
            # Normalize to sum to 1
            score_grid /= score_grid.sum()
            
            # Sum win/draw/loss probabilities
            p_home_win = np.sum(np.triu(score_grid, 1).T)  # home score > away score
            p_draw = np.sum(np.diag(score_grid))
            p_away_win = np.sum(np.tril(score_grid, -1).T)  # away score > home score
            
            # Outcome mapping: [Away Win (0), Draw (1), Home Win (2)]
            probs.append([p_away_win, p_draw, p_home_win])
            
        return np.array(probs)

def get_classifier_features() -> List[str]:
    """Returns the list of features to train the XGBoost Classifier."""
    return [
        "elo_diff", "fifa_points_diff", "fifa_rank_diff",
        "tm_val_log_ratio", "tm_val_diff",
        "home_roll_win_rate_10", "away_roll_win_rate_10",
        "home_roll_gd_10", "away_roll_gd_10",
        "home_ewa_win_rate", "away_ewa_win_rate",
        "h2h_home_win_rate", "h2h_home_gd",
        "is_home", "host_nation_boost",
        "home_manager_tenure_days", "away_manager_tenure_days",
        "home_qual_ppg", "away_qual_ppg"
    ]

def train_xgb_classifier(df_train: pd.DataFrame) -> Tuple[XGBClassifier, List[str]]:
    """
    Trains an XGBoost multiclass classifier directly on fixture outcomes.
    Outcome target is encoded as:
    - 0: Away Win (away_score > home_score)
    - 1: Draw (home_score == away_score)
    - 2: Home Win (home_score > away_score)
    """
    df = df_train.copy().dropna(subset=["home_score", "away_score"])
    
    # Target outcome encoding
    df["target"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"], df["home_score"] < df["away_score"]],
        [2, 1, 0]
    )
    
    # Feature engineering inputs
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["fifa_rank_diff"] = df["home_fifa_rank"] - df["away_fifa_rank"]
    df["fifa_points_diff"] = df["home_fifa_points"] - df["away_fifa_points"]
    df["tm_val_log_ratio"] = np.log1p(df["home_tm_total_value"]) - np.log1p(df["away_tm_total_value"])
    df["tm_val_diff"] = df["home_tm_total_value"] - df["away_tm_total_value"]
    
    features = get_classifier_features()
    df_clean = df.dropna(subset=features + ["target"])
    
    X = df_clean[features]
    y = df_clean["target"]
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        objective="multi:softprob",
        num_class=3,
        random_state=42
    )
    model.fit(X, y)
    return model, features
