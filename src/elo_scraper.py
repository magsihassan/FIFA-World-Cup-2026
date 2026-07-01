import os
import time
import requests
import pandas as pd
from typing import Dict, List, Optional

# Constants
BASE_URL = "https://www.eloratings.net"
TEAMS_LIST_URL = f"{BASE_URL}/en.teams.tsv"
RAW_DIR = os.path.join("data", "raw")
ELO_DIR = os.path.join(RAW_DIR, "eloratings")

# Ensure directories exist
os.makedirs(ELO_DIR, exist_ok=True)

def download_teams_list() -> pd.DataFrame:
    """
    Downloads the team mapping list from eloratings.net if not cached.
    Returns a DataFrame mapping country code to canonical name.
    """
    cache_path = os.path.join(RAW_DIR, "en.teams.tsv")
    if os.path.exists(cache_path):
        print("Using cached en.teams.tsv")
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"Downloading team list from {TEAMS_LIST_URL}")
        r = requests.get(TEAMS_LIST_URL)
        r.raise_for_status()
        text = r.text
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)

    # Parse using the custom tab-split logic (since rows have variable tab count)
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code, name = parts[0].strip(), parts[1].strip()
        if code.endswith("_loc"):  # Skip location strings
            continue
        rows.append({"code": code, "name": name})

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} teams from eloratings.net")
    return df

def get_url_slug(country_name: str) -> str:
    """
    Converts a country name to the URL-friendly slug used by eloratings.net.
    e.g. 'United States' -> 'United_States', 'North Korea' -> 'North_Korea'
    """
    # Replace spaces with underscores
    slug = country_name.strip().replace(" ", "_")
    return slug

def scrape_all_teams_elo(delay_seconds: float = 0.2) -> List[str]:
    """
    Scrapes the historical Elo ratings (.tsv files) for all teams in the teams list.
    Saves them to data/raw/eloratings/{country_slug}.tsv.
    Returns list of successfully scraped slugs.
    """
    teams_df = download_teams_list()
    successful_slugs = []

    for idx, row in teams_df.iterrows():
        code = row["code"]
        name = row["name"]
        slug = get_url_slug(name)
        file_path = os.path.join(ELO_DIR, f"{slug}.tsv")

        if os.path.exists(file_path):
            successful_slugs.append(slug)
            continue

        url = f"{BASE_URL}/{slug}.tsv"
        print(f"Scraping {name} ({code}) from {url}...")
        try:
            r = requests.get(url)
            if r.status_code == 200:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                successful_slugs.append(slug)
            elif r.status_code == 404:
                # Some teams listed in en.teams.tsv might not have individual .tsv files
                print(f"Skipping {name} (404 - no individual TSV page)")
            else:
                print(f"Failed to scrape {name} (Status code: {r.status_code})")
            
            # Rate limiting
            time.sleep(delay_seconds)
        except Exception as e:
            print(f"Error scraping {name}: {e}")
            time.sleep(delay_seconds)

    print(f"Finished scraping. Successfully retrieved {len(successful_slugs)} team files.")
    return successful_slugs

def parse_country_tsv(file_path: str) -> pd.DataFrame:
    """
    Parses a single country's Elo rating history file.
    Returns a DataFrame containing the match history with Elo ratings.
    """
    columns = [
        "year", "month", "day", "team_a", "team_b", 
        "team_a_score", "team_b_score", "tournament", "host", 
        "points", "team_a_rating", "team_b_rating", 
        "team_a_rank_change", "team_b_rank_change", 
        "team_a_rank", "team_b_rank"
    ]
    
    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    rows = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        # If there are fewer columns, pad with empty strings
        if len(parts) < len(columns):
            parts += [""] * (len(columns) - len(parts))
        elif len(parts) > len(columns):
            parts = parts[:len(columns)]
        rows.append(parts)
        
    df = pd.DataFrame(rows, columns=columns)
    
    # Replace unicode minus sign (\u2212) with standard hyphen
    unicode_minus = "\u2212"
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(unicode_minus, "-", regex=False)
        
    # Convert numeric fields
    numeric_cols = [
        "year", "month", "day", "team_a_score", "team_b_score", 
        "points", "team_a_rating", "team_b_rating",
        "team_a_rank_change", "team_b_rank_change",
        "team_a_rank", "team_b_rank"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # Construct a datetime column
    # Handle NaNs or invalid dates gracefully
    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]),
        errors="coerce"
    )
    
    return df

def build_elo_history_df() -> pd.DataFrame:
    """
    Builds a single master Elo history DataFrame from the scraped TSVs.
    Format:
    - date: datetime
    - team_code: str (2-letter country code)
    - rating: float
    - rank: float
    """
    teams_df = download_teams_list()
    code_to_name = dict(zip(teams_df["code"], teams_df["name"]))
    
    all_ratings = []
    
    for filename in os.listdir(ELO_DIR):
        if not filename.endswith(".tsv"):
            continue
        file_path = os.path.join(ELO_DIR, filename)
        
        try:
            df = parse_country_tsv(file_path)
            # Drop rows with invalid dates
            df = df.dropna(subset=["date"])
            
            # For each match in this country's history:
            # We record both Team A and Team B ratings AFTER the match.
            # (Note that a match between A and B will appear in BOTH A's file and B's file.
            # That's fine because we will drop duplicates later).
            for _, row in df.iterrows():
                date = row["date"]
                
                # Team A
                if pd.notna(row["team_a_rating"]):
                    all_ratings.append({
                        "date": date,
                        "team_code": row["team_a"],
                        "rating": row["team_a_rating"],
                        "rank": row["team_a_rank"]
                    })
                
                # Team B
                if pd.notna(row["team_b_rating"]):
                    all_ratings.append({
                        "date": date,
                        "team_code": row["team_b"],
                        "rating": row["team_b_rating"],
                        "rank": row["team_b_rank"]
                    })
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            
    if not all_ratings:
        return pd.DataFrame(columns=["date", "team_code", "rating", "rank"])
        
    master_df = pd.DataFrame(all_ratings)
    # Sort chronologically
    master_df = master_df.sort_values(by="date")
    # Drop duplicates for the same team on the same date, keeping the last update
    master_df = master_df.drop_duplicates(subset=["date", "team_code"], keep="last")
    
    return master_df

if __name__ == "__main__":
    # Test scraping the first 3 teams to verify logic
    teams = download_teams_list()
    print("Scraping a sample of 3 teams...")
    # Temporarily subset to test
    sample_names = teams["name"].head(3).tolist()
    for name in sample_names:
        slug = get_url_slug(name)
        url = f"{BASE_URL}/{slug}.tsv"
        r = requests.get(url)
        print(f"{name}: {r.status_code}")
