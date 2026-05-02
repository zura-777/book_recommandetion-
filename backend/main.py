from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from .recommender import HybridRecommender

app = FastAPI(title="BookWise API v2")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Recommender
recommender = HybridRecommender()
try:
    recommender.load_resources()
except Exception as e:
    print(f"Warning: Could not load resources: {e}")

class RecommendRequest(BaseModel):
    genres: List[str]
    mood: str
    last_book: str
    extra: Optional[str] = ""

@app.get("/")
def read_root():
    """Serve the updated frontend."""
    return FileResponse("frontend/index.html")

@app.post("/recommend")
async def get_hybrid_recommend(request: RecommendRequest):
    """
    Handle personalized recommendation requests.
    """
    try:
        recommendations = recommender.get_hybrid_recommendations(
            genres=request.genres,
            mood=request.mood,
            last_book=request.last_book,
            extra=request.extra
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
