"""
train.py
--------
This script TRAINS the recommendation model and SAVES it to disk.

You run this ONCE before starting the API.
After training, the model is saved as a file and the API loads it.

What is SVD?
  SVD = Singular Value Decomposition
  It's a mathematical technique for collaborative filtering.
  "Collaborative" means: instead of analyzing book content,
  we look at what other users with similar taste liked.

  Think of it like Netflix recommendations:
  "Users who liked what you liked also liked these movies."

The Surprise Library:
  'Surprise' is a Python library built specifically for
  recommendation systems. It handles the math of SVD for us.
  We just give it ratings data and it does the rest.

Run this file with:
  python train.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from models import SVDModel

# ─── Constants ─────────────────────────────────────────────────────────────────
RATINGS_PATH = os.path.join("data", "ratings.csv")    # Input: ratings data
MODEL_PATH   = os.path.join("model", "svd_model.pkl") # Output: saved model


def load_ratings_for_training() -> pd.DataFrame:
    """
    Load and prepare the ratings data for training.

    ratings.csv has 3 important columns:
      - user_id  : who rated the book
      - book_id  : which book was rated
      - rating   : score given (1 to 5)

    We'll use a sample of the data to keep training fast (for learning purposes).
    The full dataset has ~6 million ratings — takes too long for a first run.
    """
    print("[STEP 1] Loading ratings data...")

    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(
            f"Could not find '{RATINGS_PATH}'.\n"
            "Please download Goodbooks-10k from:\n"
            "  https://github.com/zygmuntz/goodbooks-10k\n"
            "and place ratings.csv in the data/ folder."
        )

    df = pd.read_csv(RATINGS_PATH)
    print(f"  → Loaded {len(df):,} total ratings")

    # Keep only the columns we need
    df = df[["user_id", "book_id", "rating"]]

    # Use a sample for faster training (remove this line to train on full data)
    # 200,000 ratings is plenty to get good results and trains in ~30 seconds
    SAMPLE_SIZE = 200_000
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)  # random_state=42 for reproducibility
        print(f"  → Using a sample of {SAMPLE_SIZE:,} ratings for faster training")

    print(f"  → Final training size: {len(df):,} ratings")
    return df


def prepare_training_data(df: pd.DataFrame):
    """
    Prepare the dataframe for our custom SVD model.
    """
    print("[STEP 2] Preparing training data...")
    return df


def train_model(df):
    """
    Train the custom SVD model on the dataset.
    """
    print("[STEP 3] Training SVD model...")
    
    model = SVDModel(n_factors=50)
    model.fit(df)

    print("  → Model training complete!")
    return model


def save_model(model):
    """
    Save the trained model to disk using pickle.

    pickle = Python's way of "serializing" (converting to bytes) any object
    so it can be saved to a file and loaded later.

    "wb" = write in binary mode (required for pickle)
    """
    print(f"[STEP 4] Saving model to {MODEL_PATH}...")

    # Make sure the model/ directory exists
    os.makedirs("model", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:   # "wb" = write binary
        pickle.dump(model, f)

    print(f"  → Model saved successfully to '{MODEL_PATH}'")


# ─── Main Training Pipeline ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   Book Recommendation System — Model Training")
    print("=" * 55)

    # Run all steps in order
    df      = load_ratings_for_training()
    model   = train_model(df)
    save_model(model)

    print()
    print("=" * 55)
    print("  Training complete! You can now run the API:")
    print("  $ uvicorn main:app --reload")
    print("=" * 55)
