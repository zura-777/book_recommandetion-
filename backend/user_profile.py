"""
backend/user_profile.py
-----------------------
Phase 2.2: Basic User Profiles

Stores per-user preference data and search history in data/profiles.json.

Profile schema
--------------
{
  "user_id":          "abc123",          # any string identifier
  "display_name":     "Alex",            # shown in the UI
  "preferred_genres": ["Fantasy", "Sci-Fi"],
  "preferred_moods":  ["Epic", "Dark"],
  "search_history": [                    # last 20 searches (FIFO)
    {
      "genres":    ["Fantasy"],
      "mood":      "Epic",
      "last_book": "Dune",
      "extra":     "",
      "timestamp": "2025-07-01T10:30:00+00:00"
    }
  ],
  "created_at": "2025-07-01T10:00:00+00:00",
  "updated_at": "2025-07-01T10:30:00+00:00"
}

Design notes
------------
- Stored in a single JSON file (data/profiles.json) — zero setup, human-readable.
- Swap for SQLite/Postgres when you need concurrent multi-worker writes.
- Profiles are created lazily (auto-created on first interaction if needed).
- Search history is capped at 20 entries (oldest dropped first).
"""

import os
import json
from datetime import datetime, timezone
from typing import Optional

PROFILES_PATH    = os.path.join("data", "profiles.json")
MAX_HISTORY_SIZE = 20  # keep only the last N searches per user


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _load_profiles() -> dict:
    if not os.path.exists(PROFILES_PATH):
        return {}
    try:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_profiles(profiles: dict) -> None:
    os.makedirs(os.path.dirname(PROFILES_PATH) or ".", exist_ok=True)
    tmp = PROFILES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    os.replace(tmp, PROFILES_PATH)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_empty_profile(user_id: str, display_name: str = "") -> dict:
    now = _now_iso()
    return {
        "user_id":          str(user_id),
        "display_name":     display_name or f"Reader #{user_id[:8]}",
        "preferred_genres": [],
        "preferred_moods":  [],
        "search_history":   [],
        "created_at":       now,
        "updated_at":       now,
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def create_profile(
    user_id:      str,
    display_name: str  = "",
    genres:       list = None,
    moods:        list = None,
) -> dict:
    """
    Create a new user profile and persist it.

    If a profile already exists for this user_id, return the existing one
    unchanged (idempotent — safe to call on every login).

    Parameters
    ----------
    user_id      : unique string identifier (email, UUID, etc.)
    display_name : optional human-readable name
    genres       : initial preferred genres (can be updated later)
    moods        : initial preferred moods  (can be updated later)

    Returns
    -------
    The new (or existing) profile dict.
    """
    profiles = _load_profiles()
    uid      = str(user_id)

    if uid in profiles:
        return profiles[uid]  # Already exists — don't overwrite

    profile = _make_empty_profile(uid, display_name)
    if genres:
        profile["preferred_genres"] = list(genres)
    if moods:
        profile["preferred_moods"] = list(moods)

    profiles[uid] = profile
    _save_profiles(profiles)
    return profile


def get_profile(user_id: str) -> Optional[dict]:
    """
    Retrieve a user's profile.

    Returns the profile dict, or None if this user_id has never been seen.
    Callers should handle None (e.g. prompt user to create a profile).
    """
    return _load_profiles().get(str(user_id))


def get_or_create_profile(user_id: str, display_name: str = "") -> dict:
    """
    Get the profile if it exists; create and return a new one if not.
    Convenience wrapper used by the /recommend endpoint.
    """
    profile = get_profile(user_id)
    if profile is None:
        profile = create_profile(user_id=user_id, display_name=display_name)
    return profile


def update_profile(
    user_id:      str,
    display_name: str  = None,
    genres:       list = None,
    moods:        list = None,
) -> dict:
    """
    Update mutable fields on a user profile.

    Only non-None arguments are applied — pass None to leave a field unchanged.

    Parameters
    ----------
    user_id      : user to update
    display_name : new display name (optional)
    genres       : new preferred genres list, replaces existing (optional)
    moods        : new preferred moods list,  replaces existing (optional)

    Returns
    -------
    The updated profile dict.
    """
    profiles = _load_profiles()
    uid      = str(user_id)

    if uid not in profiles:
        profiles[uid] = _make_empty_profile(uid)

    profile = profiles[uid]

    if display_name is not None:
        profile["display_name"] = display_name
    if genres is not None:
        profile["preferred_genres"] = list(genres)
    if moods is not None:
        profile["preferred_moods"] = list(moods)

    profile["updated_at"] = _now_iso()

    profiles[uid] = profile
    _save_profiles(profiles)
    return profile


def add_search_to_history(user_id: str, search: dict) -> dict:
    """
    Append a search to a user's history (auto-creates profile if needed).

    The search dict should contain the fields from RecommendRequest:
        { "genres": [...], "mood": "...", "last_book": "...", "extra": "..." }

    A timestamp is added automatically. History is capped at MAX_HISTORY_SIZE
    entries; the oldest entry is dropped when the cap is exceeded.

    Parameters
    ----------
    user_id : str  — user identifier
    search  : dict — the search parameters (without user_id or timestamp)

    Returns
    -------
    The updated profile dict.
    """
    profiles = _load_profiles()
    uid      = str(user_id)

    if uid not in profiles:
        profiles[uid] = _make_empty_profile(uid)

    profile = profiles[uid]

    entry = {
        "genres":    search.get("genres", []),
        "mood":      search.get("mood", ""),
        "last_book": search.get("last_book", ""),
        "extra":     search.get("extra", ""),
        "timestamp": _now_iso(),
    }

    history = profile.get("search_history", [])
    history.append(entry)

    # Keep only the most recent MAX_HISTORY_SIZE searches
    profile["search_history"] = history[-MAX_HISTORY_SIZE:]
    profile["updated_at"]     = _now_iso()

    # Auto-update preferred genres/moods based on search patterns
    _maybe_update_preferences_from_search(profile, entry)

    profiles[uid] = profile
    _save_profiles(profiles)
    return profile


def _maybe_update_preferences_from_search(profile: dict, search: dict) -> None:
    """
    Lightweight implicit preference learning.

    If a genre or mood appears in 3+ recent searches, it's added to
    preferred_genres / preferred_moods automatically.

    This runs in-place on the profile dict (no separate persistence call).
    """
    recent  = profile.get("search_history", [])[-10:]  # look at last 10 searches

    # Count genre frequencies
    genre_counts: dict = {}
    for s in recent:
        for g in s.get("genres", []):
            genre_counts[g] = genre_counts.get(g, 0) + 1

    # Count mood frequencies
    mood_counts: dict = {}
    for s in recent:
        m = s.get("mood", "")
        if m:
            mood_counts[m] = mood_counts.get(m, 0) + 1

    THRESHOLD = 3  # must appear in this many searches to be considered a preference

    current_genres = set(profile.get("preferred_genres", []))
    current_moods  = set(profile.get("preferred_moods",  []))

    for genre, count in genre_counts.items():
        if count >= THRESHOLD:
            current_genres.add(genre)

    for mood, count in mood_counts.items():
        if count >= THRESHOLD:
            current_moods.add(mood)

    profile["preferred_genres"] = list(current_genres)
    profile["preferred_moods"]  = list(current_moods)


def delete_profile(user_id: str) -> bool:
    """
    Permanently delete a user's profile.

    Returns True if a profile was deleted, False if user_id was not found.
    """
    profiles = _load_profiles()
    uid      = str(user_id)

    if uid not in profiles:
        return False

    del profiles[uid]
    _save_profiles(profiles)
    return True


def get_all_profiles() -> list:
    """
    Return all stored profiles as a list of dicts.
    Useful for admin views or analytics.
    """
    return list(_load_profiles().values())
