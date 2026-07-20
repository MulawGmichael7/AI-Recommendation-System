"""
Hybrid recommender engine for Meme Trading PLC.

Combines two signals:

1. COLLABORATIVE FILTERING (matrix factorization via truncated SVD on an
   implicit feedback matrix built from views/purchases/trades/wishlists,
   weighted by event type and blended with explicit ratings where present).
   Captures "users who liked similar memes also liked X".

2. CONTENT-BASED FILTERING (TF-IDF over item category + tags), capturing
   "this item is similar in content to items you already liked" — this
   is what solves cold-start for brand-new listings with no interactions yet.

Final score for a candidate item = alpha * CF_score + (1 - alpha) * CB_score,
alpha tunable per user based on how much interaction history they have
(new users lean content-based; established users lean collaborative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

EVENT_WEIGHTS = {
    "view": 1.0,
    "wishlist": 2.0,
    "purchase": 4.0,
    "trade": 4.0,
}


@dataclass
class RecommenderConfig:
    n_svd_components: int = 32
    min_interactions_for_cf: int = 3   # below this, user relies on content-based
    cf_full_confidence_at: int = 15    # at/above this many interactions, alpha maxes out
    max_alpha: float = 0.8             # cap on collaborative-filtering weight


class MemeRecommender:
    def __init__(self, config: RecommenderConfig | None = None):
        self.config = config or RecommenderConfig()
        self.user_ids_: list[int] = []
        self.item_ids_: list[int] = []
        self.user_index_: dict[int, int] = {}
        self.item_index_: dict[int, int] = {}
        self.user_factors_: np.ndarray | None = None
        self.item_factors_cf_: np.ndarray | None = None
        self.item_factors_cb_: np.ndarray | None = None
        self.tfidf_vectorizer_: TfidfVectorizer | None = None
        self.items_df_: pd.DataFrame | None = None
        self.user_seen_items_: dict[int, set] = {}

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, users_df: pd.DataFrame, items_df: pd.DataFrame, interactions_df: pd.DataFrame):
        self.items_df_ = items_df.reset_index(drop=True)
        self.user_ids_ = sorted(users_df.user_id.unique().tolist())
        self.item_ids_ = sorted(items_df.item_id.unique().tolist())
        self.user_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_index_ = {it: i for i, it in enumerate(self.item_ids_)}

        self._fit_collaborative(interactions_df)
        self._fit_content_based(items_df)
        self._track_seen_items(interactions_df)
        return self

    def _fit_collaborative(self, interactions_df: pd.DataFrame):
        n_users, n_items = len(self.user_ids_), len(self.item_ids_)

        rows, cols, vals = [], [], []
        for r in interactions_df.itertuples():
            if r.user_id not in self.user_index_ or r.item_id not in self.item_index_:
                continue
            u = self.user_index_[r.user_id]
            i = self.item_index_[r.item_id]
            weight = EVENT_WEIGHTS.get(r.event_type, 1.0)
            if pd.notna(r.rating):
                # blend explicit rating (1-5) in, on top of the implicit event weight
                weight += float(r.rating)
            rows.append(u)
            cols.append(i)
            vals.append(weight)

        interaction_matrix = csr_matrix(
            (vals, (rows, cols)), shape=(n_users, n_items)
        )
        # sum duplicate (user,item) entries automatically via csr_matrix construction
        interaction_matrix.sum_duplicates()

        n_components = min(self.config.n_svd_components, min(n_users, n_items) - 1)
        n_components = max(n_components, 2)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors_ = svd.fit_transform(interaction_matrix)
        self.item_factors_cf_ = svd.components_.T  # (n_items, n_components)

        self._interaction_counts = np.asarray((interaction_matrix > 0).sum(axis=1)).flatten()

    def _fit_content_based(self, items_df: pd.DataFrame):
        text = (
            items_df["category"].str.replace("-", " ")
            + " "
            + items_df["tags"].str.replace("|", " ").str.replace("-", " ")
            + " "
            + items_df["rarity"]
        )
        self.tfidf_vectorizer_ = TfidfVectorizer()
        tfidf_matrix = self.tfidf_vectorizer_.fit_transform(text)
        self.item_factors_cb_ = normalize(tfidf_matrix).toarray()

    def _track_seen_items(self, interactions_df: pd.DataFrame):
        self.user_seen_items_ = {}
        for uid, grp in interactions_df.groupby("user_id"):
            self.user_seen_items_[uid] = set(grp.item_id.tolist())

    # ------------------------------------------------------------------ #
    # Scoring / recommending
    # ------------------------------------------------------------------ #
    def _alpha_for_user(self, user_id: int) -> float:
        """How much to trust collaborative filtering vs content-based for this user."""
        if user_id not in self.user_index_:
            return 0.0  # brand new user -> pure content-based / popularity fallback
        idx = self.user_index_[user_id]
        n_int = self._interaction_counts[idx]
        cfg = self.config
        if n_int < cfg.min_interactions_for_cf:
            return 0.15
        frac = min(1.0, n_int / cfg.cf_full_confidence_at)
        return cfg.max_alpha * frac

    def _cf_scores(self, user_id: int) -> np.ndarray:
        if user_id not in self.user_index_:
            return np.zeros(len(self.item_ids_))
        u = self.user_index_[user_id]
        uvec = self.user_factors_[u]
        return self.item_factors_cf_ @ uvec

    def _cb_scores(self, user_id: int) -> np.ndarray:
        """Score items by similarity to items the user has already liked (purchased/traded/wishlisted)."""
        liked = self.user_seen_items_.get(user_id, set())
        liked_idx = [self.item_index_[i] for i in liked if i in self.item_index_]
        if not liked_idx:
            return np.zeros(len(self.item_ids_))
        profile = self.item_factors_cb_[liked_idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(profile, self.item_factors_cb_).flatten()
        return sims

    def _popularity_scores(self) -> np.ndarray:
        counts = np.zeros(len(self.item_ids_))
        # fallback for total cold-start users with zero history: use item popularity
        idx_lookup = self.item_index_
        for uid, items in self.user_seen_items_.items():
            for it in items:
                if it in idx_lookup:
                    counts[idx_lookup[it]] += 1
        if counts.max() > 0:
            counts = counts / counts.max()
        return counts

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
        category_filter: str | None = None,
    ) -> pd.DataFrame:
        """Return top-k recommended items for a user with score breakdown."""
        cf = self._cf_scores(user_id)
        cb = self._cb_scores(user_id)
        alpha = self._alpha_for_user(user_id)

        def _norm(x):
            rng = x.max() - x.min()
            return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

        cf_n, cb_n = _norm(cf), _norm(cb)

        if user_id not in self.user_index_ or (cf.sum() == 0 and cb.sum() == 0):
            # fully cold-start: fall back to popularity
            scores = self._popularity_scores()
        else:
            scores = alpha * cf_n + (1 - alpha) * cb_n

        result = self.items_df_.copy()
        result["score"] = scores
        result["cf_score"] = cf_n
        result["cb_score"] = cb_n
        result["alpha_used"] = alpha

        if exclude_seen:
            seen = self.user_seen_items_.get(user_id, set())
            result = result[~result.item_id.isin(seen)]

        if category_filter:
            result = result[result.category == category_filter]

        result = result.sort_values("score", ascending=False).head(k)
        return result.reset_index(drop=True)

    def similar_items(self, item_id: int, k: int = 10) -> pd.DataFrame:
        """Content-based 'more like this' — useful on a product detail page."""
        if item_id not in self.item_index_:
            raise ValueError(f"Unknown item_id {item_id}")
        idx = self.item_index_[item_id]
        sims = cosine_similarity(
            self.item_factors_cb_[idx:idx + 1], self.item_factors_cb_
        ).flatten()
        result = self.items_df_.copy()
        result["similarity"] = sims
        result = result[result.item_id != item_id]
        return result.sort_values("similarity", ascending=False).head(k).reset_index(drop=True)
