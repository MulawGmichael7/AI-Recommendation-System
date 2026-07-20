"""
Simple offline evaluation: leave-one-out style split.

For each user with >= 5 positive interactions (purchase/trade/wishlist),
hold out their most recent positive interaction, train on the rest, and
check whether the held-out item appears in the model's top-K recommendations.

Reports Precision@K and Recall@K (macro-averaged over users) plus
Hit Rate@K (fraction of users for whom the held-out item was recovered at all).

Run:
    python src/evaluate.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

from recommender import MemeRecommender, RecommenderConfig

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

POSITIVE_EVENTS = {"purchase", "trade", "wishlist"}
K = 10


def main():
    users_df = pd.read_csv(DATA_DIR / "users.csv")
    items_df = pd.read_csv(DATA_DIR / "items.csv")
    interactions_df = pd.read_csv(DATA_DIR / "interactions.csv")

    positives = interactions_df[interactions_df.event_type.isin(POSITIVE_EVENTS)]
    counts = positives.groupby("user_id").size()
    eligible_users = counts[counts >= 5].index.tolist()

    print(f"Evaluating on {len(eligible_users)} users with >=5 positive interactions")

    # hold out the last positive interaction (by 'day') per eligible user
    holdout_rows = []
    train_interactions = interactions_df.copy()
    for uid in eligible_users:
        user_pos = positives[positives.user_id == uid].sort_values("day")
        holdout = user_pos.iloc[-1]
        holdout_rows.append(holdout)
        # remove this exact row from training data
        mask = ~(
            (train_interactions.user_id == holdout.user_id)
            & (train_interactions.item_id == holdout.item_id)
            & (train_interactions.event_type == holdout.event_type)
            & (train_interactions.day == holdout.day)
        )
        train_interactions = train_interactions[mask]

    model = MemeRecommender(RecommenderConfig())
    model.fit(users_df, items_df, train_interactions)

    hits, precisions, recalls = [], [], []
    for holdout in holdout_rows:
        uid, target_item = holdout.user_id, holdout.item_id
        recs = model.recommend(uid, k=K, exclude_seen=True)
        rec_items = set(recs.item_id.tolist())
        hit = target_item in rec_items
        hits.append(hit)
        precisions.append((1 if hit else 0) / K)
        recalls.append(1 if hit else 0)  # single held-out item -> recall is 0/1

    print(f"\nResults @K={K}")
    print(f"  Hit Rate@{K}:   {np.mean(hits):.3f}")
    print(f"  Precision@{K}:  {np.mean(precisions):.3f}")
    print(f"  Recall@{K}:     {np.mean(recalls):.3f}")
    print(f"\n(For reference, random guessing baseline Hit Rate@{K} \u2248 {K / len(items_df):.3f})")


if __name__ == "__main__":
    main()
