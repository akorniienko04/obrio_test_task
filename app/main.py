import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.collector import CollectionResult, CollectRequest, collect_reviews
from app.insights import InsightsError, generate_insights
from app.storage import collection_directory, save_collection

app = FastAPI(
    title="App Store Review Collection API",
    description="Collect and clean a random sample of Apple App Store reviews.",
    version="0.1.0",
)
DATA_ROOT = Path("data").resolve()


class AnalyzeRequest(BaseModel):
    collection_id: str


class CollectResponse(CollectionResult):
    collection_id: str
    reviews_json: str
    reviews_csv: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reviews/collect", response_model=CollectResponse)
async def collect(request: CollectRequest) -> CollectResponse:
    try:
        result = await collect_reviews(request)
        collection_id, _ = save_collection(result, DATA_ROOT)
        return CollectResponse(
            **result.model_dump(),
            collection_id=collection_id,
            reviews_json=f"/reviews/{collection_id}/download?format=json",
            reviews_csv=f"/reviews/{collection_id}/download?format=csv",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/reviews/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """Run sentiment and rating/keyword analysis for a collected JSON file."""
    from scripts.analyze_sentiment import add_sentiment

    try:
        directory = collection_directory(request.collection_id, DATA_ROOT)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Collection was not found") from exc
    input_path = directory / "reviews.json"
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        reviews = [review for review in payload.get("reviews", []) if isinstance(review, dict)]
        from transformers import pipeline

        texts = [str(review.get("cleaned_text", "")) for review in reviews]
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Every review must contain a non-empty cleaned_text")
        predictions = pipeline(
            "sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )(texts, batch_size=16, truncation=True)
        result = add_sentiment(payload, predictions)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    output_path = directory / "analysis.json"
    try:
        result.update(generate_insights(result).model_dump())
    except InsightsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["collection_id"] = request.collection_id
    result["analysis_file"] = str(output_path.relative_to(DATA_ROOT.parent))
    return result


@app.get("/reviews/{collection_id}/download")
def download_reviews(collection_id: str, format: Literal["json", "csv"] = "json") -> FileResponse:
    """Download a collected raw JSON or CSV file from data/."""
    try:
        path = collection_directory(collection_id, DATA_ROOT) / f"reviews.{format}"
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Collection was not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Review file was not found")
    media_type = "application/json" if format == "json" else "text/csv"
    return FileResponse(path, filename=path.name, media_type=media_type)
