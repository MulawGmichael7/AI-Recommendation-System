"""
Trains the MemeRecommender on data/users.csv, data/items.csv, data/interactions.csv
and pickles the fitted model to models/recommender.pkl.

Run:
    python src/train.py
"""

import pickle
from pathlib import Path
import pandas as pd

from recommender import MemeRecommender, RecommenderConfig

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def main():
    users_df = pd.read_csv(DATA_DIR / "users.csv")
    items_df = pd.read_csv(DATA_DIR / "items.csv")
    interactions_df = pd.read_csv(DATA_DIR / "interactions.csv")

    print(f"Loaded {len(users_df)} users, {len(items_df)} items, {len(interactions_df)} interactions")

    model = MemeRecommender(RecommenderConfig())
    model.fit(users_df, items_df, interactions_df)

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "recommender.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to {MODEL_DIR / 'recommender.pkl'}")

    # quick sanity check
    sample_user = users_df.user_id.iloc[0]
    recs = model.recommend(sample_user, k=5)
    print(f"\nSample recommendations for user {sample_user}:")
    print(recs[["item_id", "title", "category", "rarity", "price_usd", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
