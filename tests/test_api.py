import importlib
import sys
from types import SimpleNamespace

import httpx

from app.collector import AppInfo, CollectionResult, Review

main = importlib.import_module("app.main")


def sample_collection() -> CollectionResult:
    return CollectionResult(
        app=AppInfo(id="123", name="Example", country="us"),
        requested_count=1,
        collected_count=1,
        available_pool_size=1,
        is_partial=False,
        reviews=[
            Review(
                id="review-1",
                title="Bad billing",
                text="Charged after free trial",
                rating=1,
                country="us",
                cleaned_text="Bad billing. Charged after free trial",
            )
        ],
    )


async def test_linked_collect_analyze_download_flow(tmp_path, monkeypatch) -> None:
    async def fake_collect(_request):
        return sample_collection()

    monkeypatch.setattr(main, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(main, "collect_reviews", fake_collect)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            pipeline=lambda *args, **kwargs: (
                lambda texts, **options: [{"label": "negative", "score": 0.98} for _ in texts]
            )
        ),
    )
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        collected = await client.post("/reviews/collect", json={"app": "123", "count": 1})
        assert collected.status_code == 200
        collection_id = collected.json()["collection_id"]
        assert (tmp_path / collection_id / "reviews.json").is_file()
        assert (tmp_path / collection_id / "reviews.csv").is_file()

        analyzed = await client.post("/reviews/analyze", json={"collection_id": collection_id})
        assert analyzed.status_code == 200
        assert analyzed.json()["rating_metrics"]["average_rating"] == 1.0
        assert (tmp_path / collection_id / "analysis.json").is_file()

        downloaded = await client.get(f"/reviews/{collection_id}/download?format=json")
        assert downloaded.status_code == 200
        assert downloaded.json()["reviews"][0]["id"] == "review-1"
