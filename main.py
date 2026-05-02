"""
main.py
-------
BookWise API — v2.1 (Phase 2: Google Books + User Profiles)

New in this version:
  - POST /recommend now enriches results with real cover images,
    descriptions, and metadata from the Google Books API (concurrent fetch).
  - GET  /similar/{book_id} also enriches results with covers/descriptions.
  - POST /users              — create a user profile
  - GET  /users/{user_id}   — get a user's profile
  - PATCH /users/{user_id}  — update preferences (genres, moods, display name)
  - DELETE /users/{user_id} — delete a profile
  - Search history is automatically saved per user on every /recommend call.

Environment variables:
  GOOGLE_BOOKS_API_KEY — optional; improves Google Books rate limits.
  Place in a .env file and install python-dotenv to load automatically.

Run:
    uvicorn main:app --reload
"""

# Load .env file if present (GOOGLE_BOOKS_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from backend.recommender  import HybridRecommender
from backend.feedback     import add_feedback, get_feedback, remove_feedback
from backend.google_books import enrich_recommendations, enrich_similar_books
from backend.user_profile import (
    create_profile, get_profile, get_or_create_profile,
    update_profile, delete_profile, add_search_to_history,
)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="📚 BookWise API",
    description=(
        "A hybrid book recommendation system combining "
        "Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF)."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Recommender (loaded once at startup) ─────────────────────────────────────
recommender = HybridRecommender()
try:
    recommender.load_resources()
    print("[STARTUP] OK: HybridRecommender ready.")
except Exception as e:
    # Server still starts; endpoints will return 503 until resources are loaded
    print(f"[STARTUP] WARNING: Could not load recommender resources: {e}")


# ─── Request / Response Schemas ───────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query:     Optional[str] = ""
    genres:    Optional[List[str]] = []
    mood:      Optional[str] = ""
    last_book: Optional[str] = ""
    extra:     Optional[str] = ""
    user_id:   Optional[int] = None
    n:         Optional[int] = 6


class FeedbackRequest(BaseModel):
    book_id:  int
    feedback: str # "liked" | "disliked" | "neutral"
    user_id:  Optional[str] = "anonymous"




class UndoFeedbackRequest(BaseModel):
    """Body for DELETE /feedback — remove a like/dislike."""
    user_id: str
    book_id: int


class CreateProfileRequest(BaseModel):
    """
    Body for POST /users

    Fields
    ------
    user_id      : unique identifier (email, UUID, device ID, etc.)
    display_name : human-readable name shown in the UI
    genres       : initial preferred genres (can be changed later)
    moods        : initial preferred moods  (can be changed later)
    """
    user_id:      str
    display_name: Optional[str]  = ""
    genres:       Optional[List[str]] = []
    moods:        Optional[List[str]] = []


class UpdateProfileRequest(BaseModel):
    """
    Body for PATCH /users/{user_id}

    All fields are optional — pass only the ones you want to change.
    """
    display_name: Optional[str]       = None
    genres:       Optional[List[str]] = None
    moods:        Optional[List[str]] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Frontend"])
def root():
    """Serve the frontend HTML."""
    return FileResponse("frontend/index.html")


@app.get("/search", tags=["Search"])
async def search_books(q: str):
    """
    Autocomplete search for books.
    """
    if recommender.books_df is None:
        return {"results": []}
    results = recommender.search_books(q)
    return {"results": results}


@app.post("/recommend", tags=["Recommendations"])
async def get_recommendations(request: RecommendRequest):
    """
    POST /recommend

    Returns top-N hybrid recommendations enriched with Google Books data.

    Pipeline:
      1. Pull user feedback (liked/disliked book_ids) if user_id provided.
      2. Run HybridRecommender (0.6 × SVD + 0.4 × TF-IDF) to get raw results.
      3. Concurrently fetch cover images, real descriptions, and metadata
         from Google Books API for each result (~200 ms for 6 books).
      4. If user_id provided: save this search to the user's history and
         auto-update their implicit preferences (genres/moods appearing 3+
         times in recent history are added to preferred lists).

    Scoring formula:
        Final = 0.6 × SVD_score(normalised 0–1) + 0.4 × TF-IDF_cosine
    """
    if recommender.books_df is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not ready — resources failed to load at startup."
        )
    try:
        # ── Step 1: Feedback ──────────────────────────────────────────────────
        feedback_data = None
        if request.user_id:
            feedback_data = get_feedback(str(request.user_id))

        # ── Step 2: Hybrid scoring ────────────────────────────────────────────
        recs = recommender.get_hybrid_recommendations(
            query         = request.query,
            genres        = request.genres,
            mood          = request.mood,
            last_book     = request.last_book,
            extra         = request.extra or "",
            user_id       = request.user_id,
            n             = request.n,
            feedback_data = feedback_data,
        )

        # ── Step 3: Google Books enrichment (concurrent) ──────────────────────
        recs = await enrich_recommendations(recs)

        # ── Step 4: Save search history ───────────────────────────────────────
        if request.user_id:
            add_search_to_history(
                user_id = str(request.user_id),
                search  = {
                    "genres":    request.genres,
                    "mood":      request.mood,
                    "last_book": request.last_book,
                    "extra":     request.extra or "",
                },
            )

        return {
            "total":           len(recs),
            "recommendations": recs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar/{book_id}", tags=["Recommendations"])
async def get_similar_books(book_id: int, n: int = 5):
    """
    GET /similar/{book_id}?n=5

    Returns N books most similar to the given book_id using
    TF-IDF cosine similarity on title/authors, enriched with
    Google Books cover images and descriptions.
    """
    if recommender.tfidf_matrix is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not ready — call load_resources() first."
        )
    try:
        similar = recommender.get_similar_books(book_id=book_id, n=n)
        if not similar:
            raise HTTPException(
                status_code=404,
                detail=f"Book with id={book_id} not found or has no similar books."
            )

        # Enrich with real covers and descriptions
        similar = await enrich_similar_books(similar)

        return {
            "book_id":       book_id,
            "similar":       similar,
            "similar_books": similar,  # Backward compatibility
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    POST /feedback

    Record a like or dislike for a book.
    This affects future recommendations for the same user_id:
      - Disliked books are excluded from results.
      - Liked books get a small score boost.
    """
    # Map "liked" -> "like", "disliked" -> "dislike"
    action = request.feedback
    if action == "liked":    action = "like"
    if action == "disliked": action = "dislike"
    
    if action == "neutral":
        result = remove_feedback(user_id=request.user_id, book_id=request.book_id)
        return {"status": "ok", **result}

    if action not in ("like", "dislike"):
        raise HTTPException(
            status_code=400,
            detail="feedback must be 'liked', 'disliked', or 'neutral'."
        )
    try:
        result = add_feedback(
            user_id = request.user_id,
            book_id = request.book_id,
            action  = action,
        )
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/{user_id}", tags=["Feedback"])
def get_user_feedback(user_id: str):
    """
    GET /feedback/{user_id}

    Retrieve the full feedback record for a user, including:
      - liked book_ids
      - disliked book_ids
      - timestamped interaction history
    """
    data = get_feedback(user_id)
    return {"user_id": user_id, **data}


@app.delete("/feedback", tags=["Feedback"])
async def undo_feedback(request: UndoFeedbackRequest):
    """
    DELETE /feedback

    Remove all like/dislike data for a specific book from a user's record.
    Useful for implementing an "undo" button in the UI.
    """
    try:
        result = remove_feedback(user_id=request.user_id, book_id=request.book_id)
        return {"status": "removed", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── User Profile Endpoints ───────────────────────────────────────────────────

@app.post("/users", tags=["User Profiles"], status_code=201)
def create_user_profile(request: CreateProfileRequest):
    """
    POST /users

    Create a new user profile. Idempotent — returns the existing profile
    if the user_id already exists (safe to call on every sign-up/login).
    """
    profile = create_profile(
        user_id      = request.user_id,
        display_name = request.display_name or "",
        genres       = request.genres or [],
        moods        = request.moods  or [],
    )
    return profile


@app.get("/users/{user_id}", tags=["User Profiles"])
def get_user_profile(user_id: str):
    """
    GET /users/{user_id}

    Retrieve a user's full profile including preferred genres, moods,
    and their last 20 searches.
    """
    profile = get_profile(user_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for user_id='{user_id}'. POST /users to create one."
        )
    return profile


@app.patch("/users/{user_id}", tags=["User Profiles"])
def patch_user_profile(user_id: str, request: UpdateProfileRequest):
    """
    PATCH /users/{user_id}

    Update one or more fields on a user's profile.
    Only the fields you pass are modified — omitted fields are unchanged.

    Example: update just the preferred genres
        PATCH /users/alice
        { "genres": ["Horror", "Thriller"] }
    """
    profile = update_profile(
        user_id      = user_id,
        display_name = request.display_name,
        genres       = request.genres,
        moods        = request.moods,
    )
    return profile


@app.delete("/users/{user_id}", tags=["User Profiles"])
def remove_user_profile(user_id: str):
    """
    DELETE /users/{user_id}

    Permanently delete a user's profile.
    Returns 404 if the user_id does not exist.
    """
    deleted = delete_profile(user_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for user_id='{user_id}'."
        )
    return {"status": "deleted", "user_id": user_id}


# ─── Dev Server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
