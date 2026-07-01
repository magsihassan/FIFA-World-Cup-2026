import os
import pandas as pd
from src.data_loader import (
    load_international_results,
    load_fifa_world_cup,
    load_fifa_rankings,
    load_transfermarkt_data
)
from src.elo_scraper import (
    download_teams_list,
    scrape_all_teams_elo,
    build_elo_history_df
)
from src.data_merger import merge_datasets
from src.name_standardizer import get_mapping_table

def main():
    print("=" * 60)
    # 1. Load basic matches and shootouts
    print("1. Loading International Matches & Shootouts...")
    fixtures_data = load_international_results()
    df_results = fixtures_data["results"]
    df_shootouts = fixtures_data["shootouts"]
    print(f"Results shape: {df_results.shape}")
    print(f"Shootouts shape: {df_shootouts.shape}")
    print("-" * 40)
    
    # 2. Run ELO Scraper
    print("2. Running ELO Scraper (from eloratings.net)...")
    teams_list_df = download_teams_list()
    # We scrape all teams with a small polite delay (0.05s is plenty fast but respectful)
    # It will use the cache if files already exist.
    scrape_all_teams_elo(delay_seconds=0.05)
    df_elo = build_elo_history_df()
    print(f"ELO History shape: {df_elo.shape}")
    print("-" * 40)
    
    # 3. Load FIFA World Cup historical matches
    print("3. Loading FIFA World Cup Matches...")
    wc_data = load_fifa_world_cup()
    df_wc_matches = wc_data["matches"]
    print(f"World Cup Matches shape: {df_wc_matches.shape}")
    print("-" * 40)
    
    # 4. Load FIFA Rankings
    print("4. Loading FIFA World Rankings...")
    df_rankings = load_fifa_rankings()
    print(f"FIFA Rankings shape: {df_rankings.shape}")
    print("-" * 40)
    
    # 5. Load Transfermarkt Data
    print("5. Loading Transfermarkt Data...")
    tm_data = load_transfermarkt_data()
    df_tm_national_teams = tm_data["national_teams"]
    print(f"Transfermarkt National Teams shape: {df_tm_national_teams.shape}")
    print("-" * 40)
    
    # Show shapes and samples of each dataset
    print("DATASETS SUMMARY:")
    for name, df in [
        ("Men's International Matches", df_results),
        ("Shootouts", df_shootouts),
        ("FIFA World Cup Matches (abecklas)", df_wc_matches),
        ("Elo Ratings History", df_elo),
        ("FIFA World Rankings", df_rankings),
        ("Transfermarkt National Teams", df_tm_national_teams)
    ]:
        print(f" - {name}: shape = {df.shape}")
        
    print("\nSAMPLE ROWS:")
    print("--- Men's International Matches Sample ---")
    print(df_results.head(2))
    print("\n--- Elo Ratings History Sample ---")
    print(df_elo.head(2))
    print("\n--- FIFA World Rankings Sample ---")
    print(df_rankings.head(2))
    print("\n--- Transfermarkt National Teams Sample ---")
    print(df_tm_national_teams.head(2))
    print("-" * 40)
    
    # Show Name-Mapping Inconsistencies Table
    print("NAME MAPPING INCONSISTENCIES / SYNONYMS:")
    mapping_tbl = get_mapping_table()
    for k, v in list(mapping_tbl.items())[:15]:
        print(f"  '{k}' -> '{v}'")
    print(f"  ... (+ {len(mapping_tbl) - 15} more mappings)")
    print("-" * 40)
    
    # 6. Merge Everything
    print("6. Merging Datasets...")
    df_merged, stats = merge_datasets(
        df_results=df_results,
        df_shootouts=df_shootouts,
        df_wc_matches=df_wc_matches,
        df_rankings=df_rankings,
        df_elo=df_elo,
        df_tm_national_teams=df_tm_national_teams,
        teams_list_df=teams_list_df
    )
    
    print("\nFINAL MERGED DATAFRAME INFO:")
    print(df_merged.info())
    print("\nFINAL MERGED DATAFRAME HEAD:")
    print(df_merged.head())
    
    print("\nPIPELINE MERGE STATISTICS:")
    print(f"Row count before merging: {stats['initial_fixtures_count']}")
    print(f"Row count after merging/cleaning: {stats['final_fixtures_count']}")
    print(f"Matches with both ELO ratings matched: {stats['matches_with_elo_both']} ({stats['matches_with_elo_both'] / stats['final_fixtures_count']:.1%})")
    print(f"Matches with both FIFA rankings matched: {stats['matches_with_fifa_both']} ({stats['matches_with_fifa_both'] / stats['final_fixtures_count']:.1%})")
    print(f"Matches with both Transfermarkt values matched: {stats['matches_with_tm_both']} ({stats['matches_with_tm_both'] / stats['final_fixtures_count']:.1%})")
    
    print("\nMISMATCH DETAILS & EXAMPLES:")
    print(f" - Elo name mismatches: {stats['elo_name_mismatches_count']} unique teams unmatched.")
    if stats['elo_name_mismatches_count'] > 0:
        print(f"   Examples: {stats['elo_name_mismatches_sample']}")
        
    print(f" - FIFA Ranking name mismatches: {stats['fifa_ranking_name_mismatches_count']} unique teams unmatched.")
    if stats['fifa_ranking_name_mismatches_count'] > 0:
        print(f"   Examples: {stats['fifa_ranking_name_mismatches_sample']}")
        
    print(f" - Transfermarkt name mismatches (expected to be higher because it tracks 124 teams): {stats['transfermarkt_name_mismatches_count']} unique teams unmatched.")
    if stats['transfermarkt_name_mismatches_count'] > 0:
        print(f"   Examples: {stats['transfermarkt_name_mismatches_sample']}")
        
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
