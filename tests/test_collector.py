import httpx
import pytest

import app.collector as collector
from app.collector import CollectRequest, clean_text, collect_reviews, extract_app_id


def test_extracts_app_id_and_cleans_text() -> None:
    assert extract_app_id("https://apps.apple.com/us/app/example/id123") == "123"
    assert clean_text(" Nice app ", "Read\nhttps://example.com   this") == "Nice app . Read this"


def test_rejects_invalid_app_id() -> None:
    with pytest.raises(ValueError):
        extract_app_id("not-an-app-store-url")


async def test_collects_deduplicated_clean_reviews(monkeypatch) -> None:
    monkeypatch.setattr(collector, "APPLE_REQUEST_DELAY_SECONDS", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        if "lookup" in str(request.url):
            return httpx.Response(200, json={"results": [{"trackName": "Example"}]})
        if request.url.params["offset"] == "0":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "1",
                            "type": "user-reviews",
                            "attributes": {"title": "Hello", "review": "Great app", "rating": 5},
                        },
                        {
                            "id": "1",
                            "type": "user-reviews",
                            "attributes": {"title": "Hello", "review": "Great app", "rating": 5},
                        },
                    ]
                },
            )
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await collect_reviews(CollectRequest(app="123", count=2, seed=1), client)

    assert result.collected_count == 1
    assert result.is_partial
    assert result.reviews[0].cleaned_text == "Hello. Great app"


async def test_rejects_empty_apple_response(monkeypatch) -> None:
    monkeypatch.setattr(collector, "APPLE_REQUEST_DELAY_SECONDS", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        if "lookup" in str(request.url):
            return httpx.Response(200, json={"results": [{"trackName": "Example"}]})
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="returned no reviews"):
            await collect_reviews(CollectRequest(app="123"), client)
