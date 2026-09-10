"""Generate actionable product insights from review analysis using an LLM."""

import json
import os
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError


class ActionableInsight(BaseModel):
    title: str
    problem: str
    evidence: list[str] = Field(default_factory=list)
    impact: str
    recommendation: str


class InsightsResult(BaseModel):
    actionable_insights: list[ActionableInsight] = Field(default_factory=list)
    status: str = "not_configured"


class InsightsError(RuntimeError):
    """Raised when the LLM cannot produce a valid insights response."""


def build_context(analysis: dict[str, Any]) -> dict[str, Any]:
    reviews = analysis.get("reviews", [])
    negative_reviews = [
        {"title": review.get("title", ""), "text": review.get("text", "")}
        for review in reviews
        if isinstance(review, dict) and review.get("sentiment") == "negative"
    ][:20]
    return {
        "app": analysis.get("app", {}),
        "rating_metrics": analysis.get("rating_metrics", {}),
        "sentiment_distribution": analysis.get("sentiment_distribution", {}),
        "negative_terms": analysis.get("negative_terms", {}),
        "representative_negative_reviews": negative_reviews,
    }


def generate_insights(analysis: dict[str, Any]) -> InsightsResult:
    if os.getenv("LLM_PROVIDER", "ollama").lower() == "ollama":
        return _generate_with_ollama(analysis)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return InsightsResult()

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = (
        "You analyze App Store review data for a product team. Return only valid JSON "
        'matching {"actionable_insights": [{"title": string, "problem": string, '
        '"evidence": [string], "impact": "high"|"medium"|"low", '
        '"recommendation": string}]}. Identify the most important recurring problems. '
        "Do not invent evidence; use only terms and reviews supplied. Return 1-5 insights."
    )
    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(build_context(analysis), ensure_ascii=False),
                    },
                ],
            },
            timeout=45.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = InsightsResult.model_validate_json(content)
        parsed.status = "generated"
        return parsed
    except (
        httpx.HTTPError,
        KeyError,
        IndexError,
        TypeError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        raise InsightsError("LLM returned an invalid or unavailable response") from exc


def _generate_with_ollama(analysis: dict[str, Any]) -> InsightsResult:
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    system_prompt = (
        "Analyze App Store review data. Return only JSON with key actionable_insights, "
        "an array of 1-5 objects containing title, problem, evidence, impact (high, medium, "
        "or low), and recommendation. Use only supplied evidence and do not invent facts."
    )
    try:
        response = httpx.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(build_context(analysis), ensure_ascii=False),
                    },
                ],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        parsed = InsightsResult.model_validate_json(content)
        parsed.status = "generated"
        return parsed
    except (
        httpx.HTTPError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        raise InsightsError(
            "Ollama is unavailable. Install Ollama and run: ollama pull llama3.2"
        ) from exc
