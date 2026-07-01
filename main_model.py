import os
import pandas as pd
from src.models import PoissonMatchModel, train_xgb_classifier
from src.evaluation import evaluate_models

FEATURED_PATH = os.path.join("data", "processed", "featured_fixtures.csv.gz")

def main():
    print("=" * 60)
    print("PHASE 3: MODEL TRAINING, EVALUATION, AND COMPARISON")
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
        print("Featured dataset successfully generated and saved.")
    else:
        print(f"Loading featured dataset from {FEATURED_PATH}...")
        df = pd.read_csv(FEATURED_PATH)
        df["date"] = pd.to_datetime(df["date"])
        
    print(f"Loaded {len(df)} matches with {len(df.columns)} columns.")
    print("-" * 40)
    
    # 2. Chronological Split (Qatar 2022 World Cup holdout)
    # The World Cup started on 2022-11-20. We set a cutoff at 2022-11-15 to separate all pre-World Cup history
    # and hold out the entire 2022 World Cup tournament for test.
    cutoff_date = "2022-11-15"
    
    df_train = df[df["date"] < cutoff_date].copy()
    df_test = df[df["date"] >= cutoff_date].copy()
    
    # Filter test set strictly to FIFA World Cup matches to make the holdout evaluation pristine
    df_test_wc = df_test[df_test["tournament"].str.contains("FIFA World Cup", case=False, na=False)].copy()
    
    print("CHRONOLOGICAL TRAIN/TEST SPLIT:")
    print(f" - Training Cutoff: Matches played before {cutoff_date}")
    print(f" - Train Set Size:  {len(df_train)} matches (Date range: {df_train['date'].min().strftime('%Y-%m-%d')} to {df_train['date'].max().strftime('%Y-%m-%d')})")
    print(f" - Test Set Size:   {len(df_test_wc)} matches (FIFA World Cup 2022, Date range: {df_test_wc['date'].min().strftime('%Y-%m-%d')} to {df_test_wc['date'].max().strftime('%Y-%m-%d')})")
    print("-" * 40)
    
    # 3. Train Models
    # A. Poisson Regression Goals Model
    print("Training Poisson Regression goals model...")
    poisson_model = PoissonMatchModel()
    poisson_model.fit(df_train)
    print("Poisson model fitted.")
    print("-" * 40)
    
    # B. XGBoost Classifier Model
    print("Training XGBoost outcome classifier...")
    xgb_model, xgb_features = train_xgb_classifier(df_train)
    print("XGBoost classifier fitted.")
    print("-" * 40)
    
    # 4. Evaluate Models
    print("4. Executing model evaluations on holdout test set...")
    metrics = evaluate_models(
        df_test=df_test_wc,
        poisson_model=poisson_model,
        xgb_model=xgb_model,
        xgb_features=xgb_features
    )
    
    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
