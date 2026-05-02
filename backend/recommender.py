import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import SVDModel

# ─── File Paths ────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("model", "svd_model.pkl")
BOOKS_PATH     = os.path.join("data", "books.csv")
TAGS_PATH      = os.path.join("data", "tags.csv")
BOOK_TAGS_PATH = os.path.join("data", "book_tags.csv")

# ─── Mood → Tag Mapping ────────────────────────────────────────────────────────
# Maps UI mood labels to Goodreads tag keywords.
MOOD_TAG_MAP = {
    "Mind-Bending":  ["philosophy", "psychological", "mind-bending",
                      "thought-provoking", "surreal", "mind-fuck", "trippy"],
    "Dark":          ["thriller", "crime", "dark", "noir",
                      "psychological-thriller", "horror", "murder", "mystery"],
    "Feel-Good":     ["feel-good", "uplifting", "heartwarming",
                      "funny", "humor", "comedy", "light-hearted"],
    "Epic":          ["epic", "adventure", "fantasy", "high-fantasy",
                      "action", "quest", "war", "saga"],
    "Romantic":      ["romance", "love", "chick-lit",
                      "contemporary-romance", "love-story"],
    "Inspiring":     ["biography", "memoir", "self-help",
                      "inspirational", "motivation", "personal-development"],
    "Classic":       ["classics", "literary-fiction", "literature",
                      "19th-century", "classic-literature", "canonical"],
    "Cozy":          ["cozy-mystery", "cozy", "slice-of-life", "gentle", "comfort"],
    # Added mappings to match UI labels
    "Light":         ["feel-good", "uplifting", "heartwarming", "funny", "humor", "light-hearted"],
    "Thought-Provoking": ["philosophy", "psychological", "thought-provoking", "surreal", "intellectual"],
    "Emotional":     ["emotional", "sad", "tear-jerker", "melancholy"],
    "Funny":         ["funny", "humor", "comedy", "laugh-out-loud"],
    "Tense":         ["tense", "suspense", "thriller", "edge-of-your-seat"],
    "Hopeful":       ["hopeful", "inspiring", "uplifting", "optimistic"],
    "Intellectual":  ["intellectual", "non-fiction", "academic", "erudite"],
    "Fast-paced":    ["fast-paced", "action", "thriller", "quick-read"],
    "Cosy":          ["cozy", "comfort", "gentle", "warm"],
}

# ─── Genre → Tag Mapping ──────────────────────────────────────────────────────
GENRE_TAG_MAP = {
    "Fantasy":    ["fantasy", "magic", "high-fantasy", "urban-fantasy", "epic-fantasy"],
    "Sci-Fi":     ["sci-fi", "science-fiction", "dystopian", "space", "cyberpunk"],
    "Mystery":    ["mystery", "crime", "detective", "whodunit"],
    "Thriller":   ["thriller", "suspense", "psychological-thriller"],
    "Romance":    ["romance", "love", "contemporary-romance"],
    "Horror":     ["horror", "scary", "dark", "supernatural"],
    "Fiction":    ["fiction", "literary-fiction", "contemporary"],
    "Biography":  ["biography", "memoir", "autobiography"],
    "History":    ["history", "historical", "historical-fiction"],
    "Philosophy": ["philosophy", "psychology", "self-help"],
    "Poetry":     ["poetry", "poems"],
}


class HybridRecommender:
    """
    A hybrid recommendation engine combining:
      1. Collaborative Filtering (SVD) — user taste patterns
      2. Content-Based Filtering (TF-IDF cosine) — book similarity
      3. Tag-Based Filtering — mood & genre alignment
    """

    def __init__(self):
        self.model          = None
        self.books_df       = None
        self.tags_df        = None
        self.book_tags_df   = None

        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
        )
        self.tfidf_matrix   = None

        self.mood_tag_map   = MOOD_TAG_MAP
        self.genre_tag_map  = GENRE_TAG_MAP

        self._filter_cache  = {}

    def load_resources(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"SVD model not found at '{MODEL_PATH}'. Run train.py first.")
        
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        print(f"[INFO] SVD model loaded from {MODEL_PATH}")

        self.books_df = pd.read_csv(BOOKS_PATH)

        self.books_df['content'] = (
            self.books_df['title'].fillna('') + " " +
            self.books_df['authors'].fillna('') + " " +
            self.books_df.get('original_title', pd.Series([''] * len(self.books_df), index=self.books_df.index)).fillna('')
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.books_df['content'])
        print(f"[INFO] TF-IDF matrix built: {self.tfidf_matrix.shape[0]} books")

        if os.path.exists(TAGS_PATH):
            self.tags_df = pd.read_csv(TAGS_PATH)
        if os.path.exists(BOOK_TAGS_PATH):
            self.book_tags_df = pd.read_csv(BOOK_TAGS_PATH)

    def _tag_ids_for_keywords(self, keywords: list) -> list:
        if self.tags_df is None or not keywords:
            return []
        pattern = '|'.join(keywords)
        return self.tags_df[self.tags_df['tag_name'].str.contains(pattern, case=False, na=False)]['tag_id'].tolist()

    def _book_ids_for_tag_ids(self, tag_ids: list) -> set:
        if self.book_tags_df is None or not tag_ids:
            return set()
        gr_ids = self.book_tags_df[self.book_tags_df['tag_id'].isin(tag_ids)]['goodreads_book_id'].unique()
        return set(self.books_df[self.books_df['goodreads_book_id'].isin(gr_ids)]['book_id'].tolist())

    def _get_filter_ids(self, key: str, keywords: list) -> set | None:
        if key in self._filter_cache:
            return self._filter_cache[key]
        tag_ids = self._tag_ids_for_keywords(keywords)
        book_ids = self._book_ids_for_tag_ids(tag_ids)
        result = book_ids if book_ids else None
        self._filter_cache[key] = result
        return result

    def _get_mood_filter(self, mood: str) -> set | None:
        return self._get_filter_ids(key=f"mood:{mood}", keywords=self.mood_tag_map.get(mood, []))

    def _get_genre_filter(self, genres: list) -> set | None:
        all_keywords = []
        for g in genres:
            all_keywords.extend(self.genre_tag_map.get(g, [g.lower()]))
        return self._get_filter_ids(key=f"genre:{'|'.join(sorted(genres))}", keywords=all_keywords)

    def get_hybrid_recommendations(
        self,
        query:         str  = "",
        genres:        list = None,
        mood:          str  = None,
        last_book:     str  = "",
        extra:         str  = "",
        user_id:       int  = None,
        n:             int  = 6,
        feedback_data: dict = None,
    ) -> list:
        # Build a rich search string from all available inputs
        genres = genres or []
        combined_query = f"{query} {last_book} {extra} {' '.join(genres)} {mood or ''}".strip()
        
        query_vec     = self.tfidf_vectorizer.transform([combined_query])
        content_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        mood_ids  = self._get_mood_filter(mood) if mood else None
        genre_ids = self._get_genre_filter(genres) if genres else None

        if mood_ids is not None and genre_ids is not None:
            tag_filter = mood_ids | genre_ids
        else:
            tag_filter = mood_ids or genre_ids

        liked_ids    = set((feedback_data or {}).get("liked",    []))
        disliked_ids = set((feedback_data or {}).get("disliked", []))

        skip_title = (last_book or "").lower().strip()
        results    = []

        for idx, row in self.books_df.iterrows():
            book_id = int(row['book_id'])
            title   = str(row['title'])

            if skip_title and skip_title in title.lower(): continue
            if book_id in disliked_ids: continue

            c_score = float(content_scores[idx])

            if user_id is not None and self.model is not None:
                raw_svd = self.model.predict(uid=user_id, iid=book_id)
                svd_score = (raw_svd - 1) / 4.0
            else:
                svd_score = c_score

            hybrid = 0.6 * svd_score + 0.4 * c_score

            if tag_filter is not None and book_id not in tag_filter:
                hybrid *= 0.6

            if book_id in liked_ids:
                hybrid = min(1.0, hybrid * 1.15)

            final_score = int(hybrid * 100)

            results.append({
                "id":          book_id,
                "book_id":     book_id,
                "title":       title,
                "author":      str(row['authors']),
                "isbn":        str(row.get('isbn', '')),
                "isbn13":      str(row.get('isbn13', '')),
                "score":       final_score,
                "description": f"{title} is a well-known work by {row['authors']} that has been widely appreciated by readers.",
                "reason":      f"Because you like {', '.join(genres)} and {mood} style books",
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n]

    def get_similar_books(self, book_id: int, n: int = 5) -> list:
        if self.tfidf_matrix is None:
            raise RuntimeError("Resources not loaded. Call load_resources() first.")
        book_row = self.books_df[self.books_df['book_id'] == book_id]
        if book_row.empty: return []
        df_idx = book_row.index[0]
        book_vec   = self.tfidf_matrix[df_idx]
        sim_scores = cosine_similarity(book_vec, self.tfidf_matrix).flatten()
        sim_scores[df_idx] = -1.0
        top_indices = np.argsort(sim_scores)[::-1][:n]
        results = []
        for i in top_indices:
            row = self.books_df.iloc[i]
            results.append({
                "id":               int(row['book_id']),
                "book_id":          int(row['book_id']),
                "title":            str(row['title']),
                "author":           str(row['authors']),
                "score":            float(sim_scores[i]),
                "similarity_score": int(float(sim_scores[i]) * 100),
            })
        return results

    def search_books(self, query: str, limit: int = 8) -> list:
        """
        Search for books by title or author for autocomplete.
        """
        if self.books_df is None or not query:
            return []
        
        q = query.lower().strip()
        # Simple string matching for speed in autocomplete
        mask = (
            self.books_df['title'].str.lower().str.contains(q, na=False) |
            self.books_df['authors'].str.lower().str.contains(q, na=False)
        )
        matches = self.books_df[mask].head(limit)
        
        return matches['title'].tolist()
