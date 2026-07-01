import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class WorldCupSimulator:
    """
    Monte Carlo simulator for the FIFA World Cup.
    Uses a trained Poisson model to simulate match outcomes (group stage + knockouts).
    """
    def __init__(self, poisson_model, group_definitions: Dict[str, List[str]]):
        self.poisson_model = poisson_model
        self.groups = group_definitions
        # Extract features and coefficients from the trained model
        self.coefs = poisson_model.model.params.to_dict()
        self.features = poisson_model.features
        
    def _predict_lambda(self, team_feat: Dict[str, float], opp_feat: Dict[str, float], is_home: float, host_boost: float) -> float:
        """
        Calculates expected goals (lambda) for a team using the relative coefficients.
        """
        # Construct relative features
        rel_feat = {
            "const": 1.0,
            "elo_diff": team_feat["elo"] - opp_feat["elo"],
            "fifa_points_diff": team_feat["fifa_points"] - opp_feat["fifa_points"],
            "tm_val_log_ratio": np.log1p(team_feat["tm_total_value"]) - np.log1p(opp_feat["tm_total_value"]),
            "roll_win_rate_diff": team_feat["roll_win_rate_10"] - opp_feat["roll_win_rate_10"],
            "roll_gd_diff": team_feat["roll_gd_10"] - opp_feat["roll_gd_10"],
            "ewa_win_rate_diff": team_feat["ewa_win_rate"] - opp_feat["ewa_win_rate"],
            "is_home": is_home,
            "host_boost": host_boost,
            "qual_ppg_diff": team_feat["qual_ppg"] - opp_feat["qual_ppg"],
            "manager_tenure_diff": team_feat["manager_tenure_days"] - opp_feat["manager_tenure_days"]
        }
        
        # Linear combination
        linear_term = 0.0
        for f in self.features:
            if f in rel_feat:
                linear_term += self.coefs[f] * rel_feat[f]
                
        # Link function (exponential link for Poisson)
        return float(np.exp(linear_term))
        
    def simulate_match(self, team_a: str, team_b: str, team_profiles: Dict[str, Dict[str, Any]], is_knockout: bool = False) -> Tuple[int, int, str]:
        """
        Simulates a match between team A and team B.
        Returns:
            - team_a_goals: int
            - team_b_goals: int
            - winner: str (the team name that advances/wins)
        """
        profile_a = team_profiles[team_a]
        profile_b = team_profiles[team_b]
        
        # Home advantage logic (Qatar was the host of 2022 WC)
        host_boost_a = 1.0 if team_a == "Qatar" else 0.0
        host_boost_b = 1.0 if team_b == "Qatar" else 0.0
        
        # Calculate Poisson lambdas (expected goals)
        lambda_a = self._predict_lambda(profile_a, profile_b, is_home=0.0, host_boost=host_boost_a)
        lambda_b = self._predict_lambda(profile_b, profile_a, is_home=0.0, host_boost=host_boost_b)
        
        # Draw goals from Poisson distribution
        goals_a = int(np.random.poisson(lambda_a))
        goals_b = int(np.random.poisson(lambda_b))
        
        if not is_knockout:
            if goals_a > goals_b:
                return goals_a, goals_b, team_a
            elif goals_a < goals_b:
                return goals_a, goals_b, team_b
            else:
                return goals_a, goals_b, "Draw"
        else:
            # Knockout stage: resolve draws in extra time/penalties
            if goals_a > goals_b:
                return goals_a, goals_b, team_a
            elif goals_a < goals_b:
                return goals_a, goals_b, team_b
            
            # Simulate Extra Time (30 mins = 1/3 of full time lambda)
            et_goals_a = int(np.random.poisson(lambda_a / 3.0))
            et_goals_b = int(np.random.poisson(lambda_b / 3.0))
            
            total_goals_a = goals_a + et_goals_a
            total_goals_b = goals_b + et_goals_b
            
            if total_goals_a > total_goals_b:
                return total_goals_a, total_goals_b, team_a
            elif total_goals_a < total_goals_b:
                return total_goals_a, total_goals_b, team_b
                
            # If still draw, resolve by coin-flip penalty shootout
            # Penalities have high variance, model as 50/50 chance
            winner = team_a if np.random.rand() > 0.5 else team_b
            return total_goals_a, total_goals_b, winner

    def simulate_group(self, group_name: str, team_profiles: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Simulates the group stage round-robin matches for a group.
        Returns the sorted list of team names (top 2 advance).
        """
        teams = self.groups[group_name]
        
        # Standings dict: team -> {points, GD, GS, matches_played}
        standings = {t: {"points": 0, "gd": 0, "gs": 0, "h2h": {}} for t in teams}
        
        # Round-robin: 6 matches
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a = teams[i]
                team_b = teams[j]
                
                g_a, g_b, winner = self.simulate_match(team_a, team_b, team_profiles, is_knockout=False)
                
                # Update standings
                standings[team_a]["gs"] += g_a
                standings[team_b]["gs"] += g_b
                standings[team_a]["gd"] += (g_a - g_b)
                standings[team_b]["gd"] += (g_b - g_a)
                
                if winner == team_a:
                    standings[team_a]["points"] += 3
                    standings[team_a]["h2h"][team_b] = 3
                    standings[team_b]["h2h"][team_a] = 0
                elif winner == team_b:
                    standings[team_b]["points"] += 3
                    standings[team_b]["h2h"][team_a] = 3
                    standings[team_a]["h2h"][team_b] = 0
                else:
                    standings[team_a]["points"] += 1
                    standings[team_b]["points"] += 1
                    standings[team_a]["h2h"][team_b] = 1
                    standings[team_b]["h2h"][team_a] = 1
                    
        # Sort standings: Points -> GD -> GS -> H2H
        # Custom sorting logic using a tuple key
        def get_sort_key(team: str) -> Tuple[int, int, int, int]:
            s = standings[team]
            # Since Python sorts tuples element-wise, we negate values to sort descending
            return (-s["points"], -s["gd"], -s["gs"])
            
        sorted_teams = sorted(teams, key=get_sort_key)
        
        # Check for H2H tiebreaker if top 2 are tied in points, GD, and GS
        if len(sorted_teams) >= 2:
            team1, team2 = sorted_teams[0], sorted_teams[1]
            if (standings[team1]["points"] == standings[team2]["points"] and
                standings[team1]["gd"] == standings[team2]["gd"] and
                standings[team1]["gs"] == standings[team2]["gs"]):
                # check h2h result between team1 and team2
                h2h_res = standings[team1]["h2h"].get(team2, 1)
                if h2h_res == 0:  # team2 beat team1
                    # swap them
                    sorted_teams[0], sorted_teams[1] = team2, team1
                    
        return sorted_teams[:2]

    def simulate_tournament(self, team_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Runs a single full tournament simulation (Group + Knockout).
        Returns a dictionary mapping:
            - "champion": str
            - "runner_up": str
            - "third_place": str
            - "semis": List[str]
            - "quarters": List[str]
            - "r16": List[str]
        """
        # 1. Group Stage
        advancing_teams = {} # group -> [1st, 2nd]
        for g_name in self.groups.keys():
            advancing_teams[g_name] = self.simulate_group(g_name, team_profiles)
            
        r16_teams = []
        for g_name, adv in advancing_teams.items():
            r16_teams.extend(adv)
            
        # 2. Round of 16
        # Matches are structured: 1st of Group X vs 2nd of Group Y
        r16_matchups = [
            (advancing_teams["A"][0], advancing_teams["B"][1]), # R16_1
            (advancing_teams["C"][0], advancing_teams["D"][1]), # R16_2
            (advancing_teams["E"][0], advancing_teams["F"][1]), # R16_3
            (advancing_teams["G"][0], advancing_teams["H"][1]), # R16_4
            (advancing_teams["B"][0], advancing_teams["A"][1]), # R16_5
            (advancing_teams["D"][0], advancing_teams["C"][1]), # R16_6
            (advancing_teams["F"][0], advancing_teams["E"][1]), # R16_7
            (advancing_teams["H"][0], advancing_teams["G"][1])  # R16_8
        ]
        
        r16_winners = []
        for ta, tb in r16_matchups:
            _, _, winner = self.simulate_match(ta, tb, team_profiles, is_knockout=True)
            r16_winners.append(winner)
            
        # 3. Quarterfinals
        qf_matchups = [
            (r16_winners[0], r16_winners[1]), # QF1
            (r16_winners[2], r16_winners[3]), # QF2
            (r16_winners[4], r16_winners[5]), # QF3
            (r16_winners[6], r16_winners[7])  # QF4
        ]
        
        qf_winners = []
        for ta, tb in qf_matchups:
            _, _, winner = self.simulate_match(ta, tb, team_profiles, is_knockout=True)
            qf_winners.append(winner)
            
        # 4. Semifinals
        sf_matchups = [
            (qf_winners[0], qf_winners[1]), # SF1
            (qf_winners[2], qf_winners[3])  # SF2
        ]
        
        sf_winners = []
        sf_losers = []
        for ta, tb in sf_matchups:
            _, _, winner = self.simulate_match(ta, tb, team_profiles, is_knockout=True)
            sf_winners.append(winner)
            sf_losers.append(ta if winner == tb else tb)
            
        # 5. Third-place Playoff
        _, _, third_place = self.simulate_match(sf_losers[0], sf_losers[1], team_profiles, is_knockout=True)
        
        # 6. Final
        _, _, champion = self.simulate_match(sf_winners[0], sf_winners[1], team_profiles, is_knockout=True)
        runner_up = sf_winners[0] if champion == sf_winners[1] else sf_winners[1]
        
        return {
            "champion": champion,
            "runner_up": runner_up,
            "third_place": third_place,
            "semis": sf_winners + sf_losers,
            "quarters": qf_winners,
            "r16": r16_winners
        }
