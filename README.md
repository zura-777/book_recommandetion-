# 📚 Book Recommendation System

A beginner-friendly AI project that recommends books using **Collaborative Filtering** (SVD).

---

## 🗂️ Project Structure

```
book-recommender/
│
├── data/                    ← Place dataset CSVs here
│   ├── books.csv            ← 10,000 books (title, author, etc.)
│   └── ratings.csv          ← ~6 million user ratings
│
├── model/                   ← Trained model saved here (auto-created)
│   └── svd_model.pkl        ← Saved after running train.py
│
├── app/
│   ├── __init__.py          ← Makes 'app' a Python package
│   └── recommender.py       ← Core recommendation logic
│
├── main.py                  ← FastAPI web server
├── train.py                 ← Model training script
├── download_data.py         ← Auto-download the dataset
└── requirements.txt         ← Python dependencies
```

---

## 🚀 How to Run (Step by Step)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download the Dataset
```bash
python download_data.py
```
This downloads `books.csv` and `ratings.csv` into the `data/` folder automatically.

### Step 3: Train the Model
```bash
python train.py
```
This trains the SVD model and saves it as `model/svd_model.pkl`.
Training takes about **30–60 seconds** (we use 200k ratings as a sample).

### Step 4: Start the API Server
```bash
uvicorn main:app --reload
```
Server starts at: **http://localhost:8000**

### Step 5: Get Recommendations!
Open your browser or use curl:
```bash
# Get top 5 books for user #5
curl http://localhost:8000/recommend/5

# Get top 10 books for user #42
curl http://localhost:8000/recommend/42?n=10
```

**Interactive API docs (built-in):**
```
http://localhost:8000/docs
```

---

## 📦 API Reference

### `GET /recommend/{user_id}`

Get top book recommendations for a specific user.

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `user_id` | int  | path     | User ID from the dataset |
| `n`       | int  | query    | Number of books to return (default: 5) |

**Example Response:**
```json
{
  "user_id": 5,
  "total_recommendations": 5,
  "recommendations": [
    {
      "title": "The Shadow of the Wind",
      "author": "Carlos Ruiz Zafón",
      "score": 98,
      "description": "A highly engaging book by Carlos Ruiz Zafón with strong themes and compelling storytelling.",
      "reason": "Because this matches your reading pattern"
    },
    {
      "title": "The Name of the Wind",
      "author": "Patrick Rothfuss",
      "score": 95,
      "description": "A highly engaging book by Patrick Rothfuss with strong themes and compelling storytelling.",
      "reason": "Because this matches your reading pattern"
    }

  ]
}
```

---

## 🧠 How It Works

### 1. The Dataset
- **books.csv** — 10,000 books with titles, authors, ISBNs
- **ratings.csv** — millions of ratings: `user_id | book_id | rating (1–5)`

### 2. Collaborative Filtering with SVD
SVD finds hidden "taste patterns" in user ratings.

```
User A liked: Harry Potter ⭐5, Lord of the Rings ⭐5, Dune ⭐4
User B liked: Harry Potter ⭐5, Lord of the Rings ⭐4
→ Users A & B have similar fantasy taste
→ Recommend Dune to User B (they haven't read it yet)
```

SVD compresses this pattern-matching math into numbers called "latent factors".

### 3. Training (`train.py`)
1. Load ratings data
2. Convert to Surprise library format
3. Train SVD model (learns patterns from 200k ratings)
4. Save model with pickle

### 4. Prediction (`recommender.py`)
1. Load saved model
2. Find all books user hasn't rated
3. Predict score for each unrated book
4. Sort by score, return top N

### 5. The API (`main.py`)
- FastAPI receives a request: `GET /recommend/5`
- Calls `get_recommendations(user_id=5, n=5)`
- Returns JSON with book titles and predicted ratings

---

## 🐛 Troubleshooting

| Error | Fix |
|-------|-----|
| `Model not found` | Run `python train.py` first |
| `books.csv not found` | Run `python download_data.py` first |
| `No recommendations for user` | Try a different user_id (valid range: 1–53,424) |
| Slow training | Normal — using 200k ratings. Edit `SAMPLE_SIZE` in `train.py` |

---

## 📚 Dataset Info

**Goodbooks-10k** by Zygmunt Zając
- 10,000 books, ~53,000 users, ~6 million ratings
- Source: https://github.com/zygmuntz/goodbooks-10k
