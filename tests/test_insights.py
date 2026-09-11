import json

import httpx

import app.insights as insights


def sample_analysis() -> dict:
    return {
        "app": {"id": "123", "name": "Example"},
        "rating_metrics": {"average_rating": 1.0},
        "sentiment_distribution": {"negative": {"count": 1, "percentage": 100.0}},
        "negative_terms": {
            "keywords": [{"term": "billing", "count": 1}],
            "phrases": [{"phrase": "charged twice", "count": 1}],
        },
        "reviews": [
            {
                "title": "Bad billing",
                "text": "I was charged twice",
                "sentiment": "negative",
            }
        ],
    }


def test_reports_when_groq_is_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = insights.generate_insights(sample_analysis())

    assert result.status == "not_configured"
    assert result.actionable_insights == []
    assert "GROQ_API_KEY" in str(result.error)


async def test_groq_health_reports_available_model(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-20b"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await insights.check_groq_health(client)

    assert result == {
        "status": "available",
        "configured": True,
        "model": "openai/gpt-oss-20b",
        "error": None,
    }


def test_generates_schema_valid_insights_with_groq(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            content = {
                "actionable_insights": [
                    {
                        "title": "Billing issue",
                        "problem": "Users report duplicate charges",
                        "evidence": ["I was charged twice"],
                        "impact": "high",
                        "recommendation": "Add duplicate-payment detection",
                    }
                ]
            }
            return {"choices": [{"message": {"content": json.dumps(content)}}]}

    def fake_post(*args, **kwargs):
        assert args[0] == insights.GROQ_API_URL
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["json"]["response_format"]["json_schema"]["strict"] is True
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    result = insights.generate_insights(sample_analysis())

    assert result.status == "generated"
    assert result.actionable_insights[0].impact == "high"
