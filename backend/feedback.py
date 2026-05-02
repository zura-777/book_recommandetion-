import os
import json
from datetime import datetime, timezone

FEEDBACK_PATH = os.path.join("data", "feedback.json")


# ─── Private helpers ───────────────────────────────────────────────────────────

def _load_store() -> dict:
    """Read the entire feedback store from disk."""
    if not os.path.exists(FEEDBACK_PATH):
        return {}
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_store(store: dict) -> None:
    """Persist the feedback store to disk atomically-ish."""
    os.makedirs(os.path.dirname(FEEDBACK_PATH) or ".", exist_ok=True)
    # Write to a temp file first, then rename — avoids corrupting on crash
    tmp_path = FEEDBACK_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    os.replace(tmp_path, FEEDBACK_PATH)


def _empty_user_record() -> dict:
    return {"liked": [], "disliked": [], "history": []}


# ─── Public API ───────────────────────────────────────────────────────────────

def add_feedback(user_id: str, book_id: int, action: str) -> dict:
    """
    Record a like or dislike interaction for a user.

    - Automatically removes the book from the opposite list
      (e.g. liking a previously disliked book clears the dislike).
    - Appends a timestamped entry to the user's history.

    Parameters
    ----------
    user_id : str  — any string identifier for the user
    book_id : int  — internal book_id from books.csv
    action  : str  — "like" or "dislike"

    Returns
    -------
    dict with user_id, liked_count, disliked_count
    """
    if action not in ("like", "dislike"):
        raise ValueError(f"Invalid action '{action}'. Must be 'like' or 'dislike'.")

    store = _load_store()
    uid   = str(user_id)

    if uid not in store:
        store[uid] = _empty_user_record()

    record = store[uid]

    if action == "like":
        # Remove from disliked if present (user changed their mind)
        if book_id in record["disliked"]:
            record["disliked"].remove(book_id)
        # Add to liked (idempotent)
        if book_id not in record["liked"]:
            record["liked"].append(book_id)

    elif action == "dislike":
        # Remove from liked if present
        if book_id in record["liked"]:
            record["liked"].remove(book_id)
        # Add to disliked (idempotent)
        if book_id not in record["disliked"]:
            record["disliked"].append(book_id)

    # Append timestamped history entry
    record["history"].append({
        "book_id":   book_id,
        "action":    action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    _save_store(store)

    return {
        "user_id":       uid,
        "liked_count":   len(record["liked"]),
        "disliked_count": len(record["disliked"]),
    }


def get_feedback(user_id: str) -> dict:
    """
    Retrieve the feedback record for a user.

    Returns a dict with keys: liked (list), disliked (list), history (list).
    If the user has no history yet, returns empty lists.

    Parameters
    ----------
    user_id : str — the user identifier

    Returns
    -------
    dict: { "liked": [...], "disliked": [...], "history": [...] }
    """
    store = _load_store()
    return store.get(str(user_id), _empty_user_record())


def remove_feedback(user_id: str, book_id: int) -> dict:
    """
    Remove all feedback for a specific book (undo like/dislike).

    Useful for a UI "undo" button.

    Returns the updated counts.
    """
    store = _load_store()
    uid   = str(user_id)

    if uid not in store:
        return {"user_id": uid, "liked_count": 0, "disliked_count": 0}

    record = store[uid]

    record["liked"]    = [b for b in record["liked"]    if b != book_id]
    record["disliked"] = [b for b in record["disliked"] if b != book_id]

    _save_store(store)

    return {
        "user_id":        uid,
        "liked_count":    len(record["liked"]),
        "disliked_count": len(record["disliked"]),
    }


def get_all_users() -> list:
    """
    Return a list of all user_ids that have submitted feedback.
    Useful for analytics / admin views.
    """
    store = _load_store()
    return list(store.keys())
