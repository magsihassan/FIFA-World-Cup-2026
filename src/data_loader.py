import os
import shutil
import requests
import pandas as pd
import kagglehub
from typing import Dict, Optional

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Configure Kaggle credentials in env to be safe
os.environ['KAGGLE_USERNAME'] = 'hassanmagsi'
os.environ['KAGGLE_KEY'] = 'KGAT_b3b374d482281107c5a490ff69761d61'

def download_file(url: str, dest_path: str) -> None:
    """Downloads a file from a URL to a local destination path with streaming."""
    print(f"Downloading from {url} to {dest_path}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def load_international_results() -> Dict[str, pd.DataFrame]:
    """Loads historical international matches and shootouts."""
    results_path = os.path.join(RAW_DIR, "results.csv")
    shootouts_path = os.path.join(RAW_DIR, "shootouts.csv")

    if not (os.path.exists(results_path) and os.path.exists(shootouts_path)):
        try:
            print("Attempting to download international results via kagglehub...")
            path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2024")
            shutil.copy(os.path.join(path, "results.csv"), results_path)
            shutil.copy(os.path.join(path, "shootouts.csv"), shootouts_path)
            print("Successfully loaded via kagglehub.")
        except Exception as e:
            print(f"kagglehub download failed: {e}. Falling back to GitHub raw URLs...")
            download_file(
                "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
                results_path
            )
            download_file(
                "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
                shootouts_path
            )

    df_results = pd.read_csv(results_path)
    df_shootouts = pd.read_csv(shootouts_path)
    return {"results": df_results, "shootouts": df_shootouts}

def load_fifa_world_cup() -> Dict[str, pd.DataFrame]:
    """Loads FIFA World Cup historical match and squad info."""
    matches_path = os.path.join(RAW_DIR, "WorldCupMatches.csv")
    players_path = os.path.join(RAW_DIR, "WorldCupPlayers.csv")
    cups_path = os.path.join(RAW_DIR, "WorldCups.csv")

    if not (os.path.exists(matches_path) and os.path.exists(players_path) and os.path.exists(cups_path)):
        try:
            print("Attempting to download FIFA World Cup data via kagglehub...")
            path = kagglehub.dataset_download("abecklas/fifa-world-cup")
            shutil.copy(os.path.join(path, "WorldCupMatches.csv"), matches_path)
            shutil.copy(os.path.join(path, "WorldCupPlayers.csv"), players_path)
            shutil.copy(os.path.join(path, "WorldCups.csv"), cups_path)
            print("Successfully loaded via kagglehub.")
        except Exception as e:
            print(f"kagglehub download failed: {e}. Falling back to GitHub raw URLs...")
            download_file(
                "https://raw.githubusercontent.com/VIVelev/WorldCup-Prediction/master/datasets/WorldCupMatches.csv",
                matches_path
            )
            download_file(
                "https://raw.githubusercontent.com/VIVelev/WorldCup-Prediction/master/datasets/WorldCupPlayers.csv",
                players_path
            )
            download_file(
                "https://raw.githubusercontent.com/VIVelev/WorldCup-Prediction/master/datasets/WorldCups.csv",
                cups_path
            )

    # Read datasets
    df_matches = pd.read_csv(matches_path)
    df_players = pd.read_csv(players_path)
    df_cups = pd.read_csv(cups_path)
    return {"matches": df_matches, "players": df_players, "cups": df_cups}

def load_fifa_rankings() -> pd.DataFrame:
    """Loads historical official monthly FIFA rankings."""
    ranking_path = os.path.join(RAW_DIR, "fifa_ranking.csv")

    if not os.path.exists(ranking_path):
        try:
            print("Attempting to download FIFA World Ranking data via kagglehub...")
            # We use turyate/fifa-world-ranking or an equivalent
            path = kagglehub.dataset_download("turyate/fifa-world-ranking")
            # The file is typically named fifa_ranking.csv
            src_file = os.path.join(path, "fifa_ranking.csv")
            if not os.path.exists(src_file):
                # Search for any csv inside the download path
                csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
                if csv_files:
                    src_file = os.path.join(path, csv_files[0])
            shutil.copy(src_file, ranking_path)
            print("Successfully loaded via kagglehub.")
        except Exception as e:
            print(f"kagglehub download failed: {e}. Falling back to GitHub raw URL...")
            download_file(
                "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/fifa_ranking.csv",
                ranking_path
            )

    df_ranking = pd.read_csv(ranking_path)
    return df_ranking

def load_transfermarkt_data() -> Dict[str, pd.DataFrame]:
    """Loads Transfermarkt tables containing national teams, players, valuations, appearances, games, and competitions."""
    tables = ["national_teams", "players", "player_valuations", "appearances", "games", "competitions"]
    data = {}

    for table in tables:
        dest_path = os.path.join(RAW_DIR, f"{table}.csv.gz")
        if not os.path.exists(dest_path):
            try:
                print(f"Attempting to download Transfermarkt {table} via kagglehub...")
                path = kagglehub.dataset_download("davidcariboo/player-scores-database-and-creator-tool")
                src_file = os.path.join(path, f"{table}.csv")
                if os.path.exists(src_file):
                    df = pd.read_csv(src_file)
                    df.to_csv(dest_path, compression="gzip", index=False)
                else:
                    src_file_gz = os.path.join(path, f"{table}.csv.gz")
                    if os.path.exists(src_file_gz):
                        shutil.copy(src_file_gz, dest_path)
                print(f"Successfully loaded Transfermarkt {table} via kagglehub.")
            except Exception as e:
                print(f"kagglehub download failed for {table}: {e}. Falling back to Cloudflare R2 bucket URL...")
                url = f"https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/{table}.csv.gz"
                download_file(url, dest_path)

        # We load with pandas, using compression='gzip'
        data[table] = pd.read_csv(dest_path, compression="gzip")
        print(f"Loaded Transfermarkt {table}: shape = {data[table].shape}")

    return data

if __name__ == "__main__":
    print("Testing data loader...")
    results = load_international_results()
    print("Results head:")
    print(results["results"].head(2))
