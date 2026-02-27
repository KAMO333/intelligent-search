from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from MatchEngine import PropertyMatchEngine, CONFIG
import uvicorn

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Intelligent Property Search API",
    description="Hybrid Semantic + Hard Filter Search Engine",
    version="1.0.0",
)

# 2. Global Engine Instance (Pre-loads data into RAM on startup)
# Ensure your CSV is in backend/data/apartments_for_rent.csv
engine = PropertyMatchEngine(CONFIG).load_source("data/apartments_for_rent.csv")


# 3. Define the Request Body Schema
class SearchRequest(BaseModel):
    query: str
    max_price: Optional[int] = 3000
    min_bedrooms: Optional[int] = 2
    threshold: Optional[float] = 0.70


# --- ROUTES ---


@app.get("/")
async def root():
    return {"message": "AI Property Search Engine is Online", "status": "Ready"}


@app.post("/search")
async def perform_search(request: SearchRequest):
    if engine.data.empty:
        raise HTTPException(status_code=500, detail="Data source not loaded.")

    prefs = {
        "query": request.query,
        "max_price": request.max_price,
        "min_bedrooms": request.min_bedrooms,
    }

    results = engine.run_search(prefs)
    mask = results["Final_Score"] >= request.threshold

    # --- THE FIX ---
    # .fillna(0) ensures all empty numeric values are valid JSON
    top_matches = results[mask].fillna(0).head(10).to_dict(orient="records")

    return {
        "query_received": request.query,
        "total_matches_above_threshold": len(results[mask]),
        "results": top_matches,
    }


if __name__ == "__main__":
    # Run the server on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
