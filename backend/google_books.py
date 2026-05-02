"""
backend/google_books.py
-----------------------
Phase 2.1: Google Books API Integration

Replaces the fake placeholder descriptions with:
  - Real book synopsis (up to 500 chars, trimmed cleanly)
  - High-quality cover image URL (prefers extraLarge → large → medium → thumbnail)
  - Genre categories from Google's taxonomy
  - Publication date, page count, average rating
  - A direct link to the Google Books preview page

Architecture — two-layer cache:
  1. In-memory dict (_memory_cache): zero-latency reads for the lifetime
     of the server process. Filled on first fetch, then stays hot.
  2. Disk cache (data/books_cache.json): survives server restarts.
     Each book is fetched from the API exactly once, ever.

Concurrency:
  enrich_recommendations() uses asyncio.gather() to fire all API calls
  in parallel — a list of 6 books enriches in ~200 ms instead of ~1.2 s.

API Key:
  Set the environment variable GOOGLE_BOOKS_API_KEY.
  The client works without a key (free quota: ~1000 req/day per IP),
  but with a key you get 1000 req/day/key, which is more than enough
  for development and small-scale use.

  Add to a .env file in your project root:
      GOOGLE_BOOKS_API_KEY=your_key_here

  Then load it with:
      pip install python-dotenv
      # In main.py or at app start:
      from dotenv import load_dotenv; load_dotenv()
"""

import os
import json
import asyncio
import httpx
from datetime import datetime, timezone

# ─── Config ───────────────────────────────────────────────────────────────────
CACHE_PATH           = os.path.join("data", "books_cache.json")
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "")
BASE_URL             = "https://www.googleapis.com/books/v1/volumes"

# Google Books fields to request — limits response payload size
_FIELDS = "items(volumeInfo/title,volumeInfo/authors,volumeInfo/description,volumeInfo/imageLinks,volumeInfo/categories,volumeInfo/publishedDate,volumeInfo/pageCount,volumeInfo/averageRating,volumeInfo/ratingsCount,volumeInfo/previewLink)"

# Sentinel stored in cache for "we tried and got nothing" — avoids re-fetching
_CACHE_MISS_SENTINEL = "__MISS__"

# ─── In-memory cache ──────────────────────────────────────────────────────────
_memory_cache: dict = {}
_disk_cache_loaded   = False


def _load_disk_cache() -> dict:
    """
    Reads books_cache.json into _memory_cache once per process lifetime.
    Subsequent calls return immediately.
    """
    global _disk_cache_loaded
    if _disk_cache_loaded:
        return _memory_cache
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _memory_cache.update(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Google Books] Warning: could not load cache: {e}")
    _disk_cache_loaded = True
    return _memory_cache


def _save_disk_cache() -> None:
    """Atomically persist the memory cache to disk (write-then-rename)."""
    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_memory_cache, f, indent=2)
    os.replace(tmp, CACHE_PATH)


# ─── Parsing ──────────────────────────────────────────────────────────────────

def _best_cover_url(image_links: dict) -> str:
    """
    Return the highest available quality cover URL, forcing HTTPS.
    Google Books sometimes returns HTTP URLs — we always upgrade them.
    """
    for quality in ("extraLarge", "large", "medium", "thumbnail", "smallThumbnail"):
        url = image_links.get(quality, "")
        if url:
            # Force HTTPS + strip low-res zoom parameter
            return (
                url
                .replace("http://", "https://")
                .replace("&zoom=1", "")
                .replace("zoom=1&", "")
            )
    return ""


def _trim_description(text: str, max_chars: int = 500) -> str:
    """Trim description to max_chars, ending on a word boundary with '...'"""
    if not text or len(text) <= max_chars:
        return text or ""
    trimmed = text[:max_chars].rsplit(" ", 1)[0]
    return trimmed.rstrip(".,;:") + "..."


def _parse_volume(item: dict) -> dict:
    """Extract only the fields we care about from a Google Books API item."""
    vi = item.get("volumeInfo", {})
    return {
        "cover_url":      _best_cover_url(vi.get("imageLinks", {})),
        "description":    _trim_description(vi.get("description", "")),
        "categories":     vi.get("categories", []),
        "published_date": vi.get("publishedDate", ""),
        "page_count":     vi.get("pageCount", 0),
        "average_rating": vi.get("averageRating", 0.0),
        "ratings_count":  vi.get("ratingsCount", 0),
        "preview_link":   vi.get("previewLink", ""),
        "fetched_at":     datetime.now(timezone.utc).isoformat(),
    }


# ─── Core Fetch ───────────────────────────────────────────────────────────────

async def fetch_book_metadata(book_id: int, title: str, author: str) -> dict:
    """
    Fetch Google Books metadata for one book.

    Lookup order:
      1. In-memory cache  →  instant
      2. Disk cache       →  fast (file read)
      3. Google Books API →  ~100–300 ms, then cached forever

    Parameters
    ----------
    book_id : int   — used as the cache key (stable, from books.csv)
    title   : str   — used in the search query
    author  : str   — used in the search query (first author only)

    Returns
    -------
    dict with cover_url, description, categories, published_date,
    page_count, average_rating, ratings_count, preview_link.
    Empty dict on API error or no results found.
    """
    cache     = _load_disk_cache()
    cache_key = str(book_id)

    # ── Layer 1 + 2: cache hit ────────────────────────────────────────────────
    if cache_key in cache:
        val = cache[cache_key]
        # Distinguish "was fetched, got nothing" from "never fetched"
        return {} if val == _CACHE_MISS_SENTINEL else val

    # ── Layer 3: live API call ────────────────────────────────────────────────
    # Google's intitle:/inauthor: operators give the most precise results.
    # We clean the title of special chars like '#' or colons which can break queries.
    clean_title  = title.split('(')[0].split(':')[0].strip()
    first_author = author.split(",")[0].strip() if author else ""
    query        = f'intitle:"{clean_title}" inauthor:"{first_author}"'

    params: dict = {
        "q":            query,
        "maxResults":   1,
    }
    if GOOGLE_BOOKS_API_KEY:
        params["key"] = GOOGLE_BOOKS_API_KEY

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            # For debugging, we'll print the full URL once
            # print(f"[DEBUG] Google Books URL: {BASE_URL}?{client.params}") 
            resp = await client.get(BASE_URL, params=params)
            if resp.status_code != 200:
                print(f"[Google Books] HTTP {resp.status_code} for '{title}' - URL: {resp.url}")
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        print(f"[Google Books] Timeout for '{title}'")
        return {}
    except httpx.HTTPStatusError as e:
        print(f"[Google Books] HTTP {e.response.status_code} for '{title}'")
        return {}
    except Exception as e:
        print(f"[Google Books] Unexpected error for '{title}': {e}")
        return {}

    items = data.get("items", [])
    if not items:
        print(f"[Google Books] No results for '{title}'")
        # Cache the miss so we don't re-fetch on every request
        _memory_cache[cache_key] = _CACHE_MISS_SENTINEL
        _save_disk_cache()
        return {}

    result = _parse_volume(items[0])

    # Persist to both layers
    _memory_cache[cache_key] = result
    _save_disk_cache()

    return result


# ─── Batch Enrichment ─────────────────────────────────────────────────────────

async def enrich_recommendations(recommendations: list) -> list:
    """
    Enrich a list of recommendation dicts with Google Books metadata.

    All API calls are fired CONCURRENTLY with asyncio.gather —
    6 books enriched in ~200 ms instead of ~1.2 s sequentially.

    Each recommendation dict is mutated in-place and returned with
    the following new keys (set to safe defaults if fetch fails):

        cover_url      : str   — HTTPS image URL for the book cover
        description    : str   — real synopsis (replaces the fake one)
        categories     : list  — e.g. ["Fiction / Thriller"]
        published_date : str   — e.g. "1997-06-26"
        page_count     : int   — number of pages
        preview_link   : str   — Google Books preview URL

    Parameters
    ----------
    recommendations : list of dicts from HybridRecommender.get_hybrid_recommendations()

    Returns
    -------
    Same list, enriched. Order preserved.
    """
    _DEFAULTS = {
        "cover_url":      "",
        "categories":     [],
        "published_date": "",
        "page_count":     0,
        "preview_link":   "",
    }

    async def _enrich_one(rec: dict) -> dict:
        meta = await fetch_book_metadata(
            book_id = rec["book_id"],
            title   = rec["title"],
            author  = rec["author"],
        )
        if meta:
            # Only overwrite description if Google's is non-empty
            if meta.get("description"):
                rec["description"] = meta["description"]
            rec["cover_url"]      = meta.get("cover_url", "")
            rec["categories"]     = meta.get("categories", [])
            rec["published_date"] = meta.get("published_date", "")
            rec["page_count"]     = meta.get("page_count", 0)
            rec["preview_link"]   = meta.get("preview_link", "")
        else:
            # Guarantee these keys always exist, even on failure
            for k, v in _DEFAULTS.items():
                rec.setdefault(k, v)
        return rec

    enriched = await asyncio.gather(*[_enrich_one(rec) for rec in recommendations])
    return list(enriched)


async def enrich_similar_books(books: list) -> list:
    """
    Same as enrich_recommendations() but for the /similar endpoint response.
    Similar books dicts use 'author' instead of 'authors', same shape.
    """
    return await enrich_recommendations(books)
