"""
Minimal Flask API exposing the trained recommender.

Endpoints:
    GET /recommend/<user_id>?k=10&category=stonks-finance
    GET /similar/<item_id>?k=10
    GET /health

Run:
    python src/app.py
Then:
    curl http://localhost:5000/recommend/1
    curl http://localhost:5000/similar/42
"""

import pickle
from pathlib import Path
from flask import Flask, jsonify, request

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "recommender.pkl"

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "n_users": len(model.user_ids_), "n_items": len(model.item_ids_)})


@app.get("/recommend/<int:user_id>")
def recommend(user_id):
    k = request.args.get("k", default=10, type=int)
    category = request.args.get("category", default=None, type=str)
    try:
        recs = model.recommend(user_id, k=k, category_filter=category)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    cols = ["item_id", "title", "category", "rarity", "price_usd", "score"]
    return jsonify(recs[cols].to_dict(orient="records"))


@app.get("/similar/<int:item_id>")
def similar(item_id):
    k = request.args.get("k", default=10, type=int)
    try:
        sims = model.similar_items(item_id, k=k)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    cols = ["item_id", "title", "category", "rarity", "price_usd", "similarity"]
    return jsonify(sims[cols].to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
