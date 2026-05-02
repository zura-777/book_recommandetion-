import os
import pickle
import numpy as np
import pandas as pd
from backend.models import SVDModel

# Constants
RATINGS_PATH = os.path.join("data", "ratings.csv")
MODEL_PATH   = os.path.join("model", "svd_model.pkl")

def load_ratings_for_training() -> pd.DataFrame:
    print("[STEP 1] Loading ratings data...")
    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(f"Could not find '{RATINGS_PATH}'")
    df = pd.read_csv(RATINGS_PATH)
    df = df[["user_id", "book_id", "rating"]]
    SAMPLE_SIZE = 200_000
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"  → Final training size: {len(df):,} ratings")
    return df

def train_model(df):
    print("[STEP 3] Training SVD model...")
    model = SVDModel(n_factors=50)
    model.fit(df)
    print("  → Model training complete!")
    return model

def save_model(model):
    print(f"[STEP 4] Saving model to {MODEL_PATH}...")
    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  → Model saved successfully")

if __name__ == "__main__":
    df = load_ratings_for_training()
    model = train_model(df)
    save_model(model)
