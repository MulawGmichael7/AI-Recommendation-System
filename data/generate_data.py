"""
Synthetic data generator for "Meme Trading PLC" — an e-commerce platform
where users buy, sell, and trade meme items (templates, meme cards,
collectible meme packs, custom edits, etc).

Generates three CSVs:
  - users.csv        : user profiles
  - items.csv         : meme items for sale, with category/tag metadata
  - interactions.csv  : purchases / views / trades / ratings (implicit + explicit)

Run:
    python data/generate_data.py
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUT_DIR = Path(__file__).parent

N_USERS = 500
N_ITEMS = 300
N_INTERACTIONS = 12000

# --- Meme categories / formats you'd realistically see on a meme trading site ---
CATEGORIES = [
    "classic-format", "reaction", "wojak", "animal", "political-satire",
    "gaming", "anime", "deep-fried", "surreal", "stonks-finance",
    "programmer-humor", "pet-meme", "history-meme", "sports", "movie-tv",
]

TAGS_POOL = [
    "drake", "distracted-bf", "doge", "pepe", "wojak", "chad", "stonks",
    "among-us", "spongebob", "cat", "dog", "gigachad", "npc", "brain-expand",
    "this-is-fine", "surprised-pikachu", "galaxy-brain", "rare", "limited-edition",
    "hand-drawn", "ai-generated", "vintage", "trending", "og", "template",
    "custom-edit", "hd", "animated", "static", "1-of-1",
]

RARITIES = ["common", "uncommon", "rare", "epic", "legendary"]
RARITY_WEIGHTS = [0.45, 0.28, 0.16, 0.08, 0.03]
RARITY_PRICE_MULT = {"common": 1.0, "uncommon": 2.2, "rare": 5.0, "epic": 12.0, "legendary": 30.0}

USER_ARCHETYPES = ["casual-collector", "power-trader", "meme-creator", "flipper", "lurker-buyer"]


def gen_users(n):
    rows = []
    for uid in range(1, n + 1):
        archetype = random.choices(
            USER_ARCHETYPES, weights=[0.35, 0.15, 0.10, 0.15, 0.25]
        )[0]
        # each user has 1-4 favorite categories -> drives their taste
        n_fav = random.randint(1, 4)
        fav_categories = random.sample(CATEGORIES, n_fav)
        rows.append({
            "user_id": uid,
            "archetype": archetype,
            "favorite_categories": "|".join(fav_categories),
            "signup_day": random.randint(0, 365),
            "trust_score": round(np.clip(np.random.normal(75, 15), 10, 100), 1),
        })
    return pd.DataFrame(rows)


def gen_items(n):
    rows = []
    for iid in range(1, n + 1):
        category = random.choice(CATEGORIES)
        n_tags = random.randint(2, 6)
        tags = random.sample(TAGS_POOL, n_tags)
        rarity = random.choices(RARITIES, weights=RARITY_WEIGHTS)[0]
        base_price = round(np.random.uniform(0.5, 20.0), 2)
        price = round(base_price * RARITY_PRICE_MULT[rarity], 2)
        rows.append({
            "item_id": iid,
            "title": f"{category.replace('-', ' ').title()} Meme #{iid}",
            "category": category,
            "tags": "|".join(tags),
            "rarity": rarity,
            "price_usd": price,
            "creator_user_id": random.randint(1, N_USERS),
            "listed_day": random.randint(0, 365),
        })
    return pd.DataFrame(rows)


def gen_interactions(users_df, items_df, n):
    rows = []
    user_ids = users_df["user_id"].tolist()
    item_ids = items_df["item_id"].tolist()

    # map each user to their favorite categories for biased sampling
    user_fav = {
        r.user_id: set(r.favorite_categories.split("|"))
        for r in users_df.itertuples()
    }
    item_category = dict(zip(items_df.item_id, items_df.category))

    event_types = ["view", "purchase", "trade", "wishlist"]
    event_weights = [0.55, 0.20, 0.10, 0.15]

    seen = set()
    attempts = 0
    while len(rows) < n and attempts < n * 5:
        attempts += 1
        uid = random.choice(user_ids)
        favs = user_fav[uid]

        # 70% chance: bias towards a favorite category (realistic taste signal)
        if random.random() < 0.7 and favs:
            candidates = [i for i in item_ids if item_category[i] in favs]
            iid = random.choice(candidates) if candidates else random.choice(item_ids)
        else:
            iid = random.choice(item_ids)

        event = random.choices(event_types, weights=event_weights)[0]

        # explicit rating only sometimes accompanies purchase/trade
        rating = None
        if event in ("purchase", "trade") and random.random() < 0.6:
            # ratings skew positive but noisy, higher if item matches fav category
            base = 4.2 if item_category[iid] in favs else 3.3
            rating = int(np.clip(round(np.random.normal(base, 0.9)), 1, 5))

        key = (uid, iid, event)
        rows.append({
            "user_id": uid,
            "item_id": iid,
            "event_type": event,
            "rating": rating,
            "day": random.randint(0, 365),
        })

    return pd.DataFrame(rows)


def main():
    users_df = gen_users(N_USERS)
    items_df = gen_items(N_ITEMS)
    interactions_df = gen_interactions(users_df, items_df, N_INTERACTIONS)

    users_df.to_csv(OUT_DIR / "users.csv", index=False)
    items_df.to_csv(OUT_DIR / "items.csv", index=False)
    interactions_df.to_csv(OUT_DIR / "interactions.csv", index=False)

    print(f"users.csv:         {len(users_df)} rows")
    print(f"items.csv:         {len(items_df)} rows")
    print(f"interactions.csv:  {len(interactions_df)} rows")
    print(f"  purchases: {(interactions_df.event_type == 'purchase').sum()}")
    print(f"  trades:    {(interactions_df.event_type == 'trade').sum()}")
    print(f"  views:     {(interactions_df.event_type == 'view').sum()}")
    print(f"  wishlist:  {(interactions_df.event_type == 'wishlist').sum()}")
    print(f"  rated:     {interactions_df.rating.notna().sum()}")


if __name__ == "__main__":
    main()
