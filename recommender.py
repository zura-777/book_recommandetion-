"""
recommender.py
--------------
This is the BRAIN of the recommendation system.

It handles:
  - Loading the trained SVD model from disk
  - Loading the books data (titles, authors)
  - Generating top-N book recommendations for a given user

How SVD (Collaborative Filtering) works (simple explanation):
  - Every user has rated some books (e.g., user 5 rated book A: 4 stars, book B: 2 stars)
  - SVD finds hidden "taste patterns" shared across users
  - If user 5 and user 8 both liked the same books, SVD knows they have similar taste
  - It then predicts how much user 5 would like books they haven't read yet
  - We pick the top 5 predicted scores → those become recommendations
"""

import os
import pickle
import pandas as pd
from models import SVDModel


# ─── Constants (file paths) ────────────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "svd_model.pkl")        # Where the trained model lives
BOOKS_PATH = os.path.join("data", "books.csv")             # Goodbooks-10k books metadata
RATINGS_PATH = os.path.join("data", "ratings.csv")         # Goodbooks-10k user ratings


def load_model():
    """
    Load the pre-trained SVD model from disk using pickle.

    pickle is Python's built-in way to save and load any object.
    We saved the trained model as a file so we don't have to retrain every time.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Please run train.py first to train and save the model."
        )

    with open(MODEL_PATH, "rb") as f:   # "rb" = read in binary mode
        model = pickle.load(f)

    print(f"[INFO] Model loaded from {MODEL_PATH}")
    return model


def load_books() -> pd.DataFrame:
    """
    Load books.csv and return a DataFrame with book_id, title, and authors.

    DataFrame = a table (like an Excel sheet) that pandas gives us to work with.
    """
    if not os.path.exists(BOOKS_PATH):
        raise FileNotFoundError(
            f"Books data not found at '{BOOKS_PATH}'. "
            "Please download the Goodbooks-10k dataset and place books.csv in data/"
        )

    # Read the CSV file into a pandas DataFrame
    books_df = pd.read_csv(BOOKS_PATH)

    # We only need these 3 columns (keep it simple)
    books_df = books_df[["book_id", "title", "authors"]]

    print(f"[INFO] Loaded {len(books_df)} books from {BOOKS_PATH}")
    return books_df


def load_ratings() -> pd.DataFrame:
    """
    Load ratings.csv — this tells us which books each user has already rated.

    We need this so we DON'T recommend books the user has already read.
    """
    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(
            f"Ratings data not found at '{RATINGS_PATH}'. "
            "Please download the Goodbooks-10k dataset and place ratings.csv in data/"
        )

    ratings_df = pd.read_csv(RATINGS_PATH)
    print(f"[INFO] Loaded {len(ratings_df)} ratings from {RATINGS_PATH}")
    return ratings_df


def get_recommendations(user_id: int, n: int = 5, genres: list = None, mood: str = "Classic"):
    if not genres: genres = ["Compelling stories"]

    model = load_model()
    books_df = load_books()
    ratings_df = load_ratings()

    rated_books = set(
        ratings_df[ratings_df["user_id"] == user_id]["book_id"].tolist()
    )

    all_book_ids = books_df["book_id"].tolist()

    predictions = []
    for book_id in all_book_ids:
        if book_id not in rated_books:
            score = model.predict(uid=user_id, iid=book_id)
            predictions.append((book_id, score))

    # 🔥 Sort by score
    predictions.sort(key=lambda x: x[1], reverse=True)

    # 🔥 Take top N
    top_n = predictions[:n]

    results = []

    for rank, (book_id, score) in enumerate(top_n):
        book_row = books_df[books_df["book_id"] == book_id]

        if not book_row.empty:
            # ✅ Normalize score (1–5 → 0–100)
            normalized = (score - 1) / 4
            match_percent = int(normalized * 100)

            # ✅ Ranking-based boost (makes UI look better)
            rank_boost = int((n - rank) / n * 10)
            final_score = min(100, match_percent + rank_boost)

            # ✅ Fake description (replace later if needed)
            description = f"{book_row.iloc[0]['title']} is a well-known work by {book_row.iloc[0]['authors']} that has been widely appreciated by readers."


            results.append({
                "title": book_row.iloc[0]["title"],
                "author": book_row.iloc[0]["authors"],
                "score": final_score,
                "description": description,
                "reason": f"Because you like {', '.join(genres)} and {mood} style books"

            })

    return results

