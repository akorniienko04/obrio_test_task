"""Generate actionable product insights through the Groq API."""

import json
import os
import time
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

load_dotenv()


class ActionableInsight(BaseModel):
    title: str
    problem: str
    evidence: list[str] = Field(default_factory=list)
    impact: Literal["high", "medium", "low"]
    recommendation: str


class InsightsResult(BaseModel):
    actionable_insights: list[ActionableInsight] = Field(default_factory=list)
    status: Literal["generated", "not_configured", "unavailable"] = "not_configured"
    error: str | None = None


class InsightsError(RuntimeError):
    """Raised when Groq cannot produce a valid insights response."""


async def check_groq_health(
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, str | bool | None]:
    """Return Groq configuration and connectivity status without generating text."""
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "status": "not_configured",
            "configured": False,
            "model": model,
            "error": "GROQ_API_KEY is not set",
        }

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=5.0)
    try:
        response = await client.get(
            GROQ_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data", []) if isinstance(payload, dict) else []
        model_ids = {
            str(item.get("id")) for item in models if isinstance(item, dict) and item.get("id")
        }
        if model not in model_ids:
            return {
                "status": "model_unavailable",
                "configured": True,
                "model": model,
                "error": "Configured model is not available for this Groq account",
            }
        return {
            "status": "available",
            "configured": True,
            "model": model,
            "error": None,
        }
    except httpx.HTTPStatusError as exc:
        status = "invalid_credentials" if exc.response.status_code in {401, 403} else "unavailable"
        return {
            "status": status,
            "configured": True,
            "model": model,
            "error": f"Groq health request returned HTTP {exc.response.status_code}",
        }
    except (httpx.HTTPError, ValueError, TypeError):
        return {
            "status": "unavailable",
            "configured": True,
            "model": model,
            "error": "Could not reach or parse the Groq models API",
        }
    finally:
        if owns_client:
            await client.aclose()


def build_context(analysis: dict[str, Any]) -> dict[str, Any]:
    """Build a compact context that fits comfortably inside free API limits."""
    reviews = analysis.get("reviews", [])
    negative_reviews = [
        {
            "title": str(review.get("title", ""))[:200],
            "text": str(review.get("text", ""))[:600],
        }
        for review in reviews
        if isinstance(review, dict) and review.get("sentiment") == "negative"
    ][:12]

    negative_terms = analysis.get("negative_terms", {})
    if not isinstance(negative_terms, dict):
        negative_terms = {}

    return {
        "app": analysis.get("app", {}),
        "rating_metrics": analysis.get("rating_metrics", {}),
        "sentiment_distribution": analysis.get("sentiment_distribution", {}),
        "negative_terms": {
            "keywords": negative_terms.get("keywords", [])[:15],
            "phrases": negative_terms.get("phrases", [])[:15],
        },
        "representative_negative_reviews": negative_reviews,
    }


def _response_schema() -> dict[str, Any]:
    insight_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "problem": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "impact": {"type": "string", "enum": ["high", "medium", "low"]},
            "recommendation": {"type": "string"},
        },
        "required": ["title", "problem", "evidence", "impact", "recommendation"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "app_store_actionable_insights",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"actionable_insights": {"type": "array", "items": insight_schema}},
                "required": ["actionable_insights"],
                "additionalProperties": False,
            },
        },
    }


def generate_insights(analysis: dict[str, Any]) -> InsightsResult:
    """Generate insights with Groq or report that no API key was configured."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return InsightsResult(error="Set GROQ_API_KEY to enable actionable insights")

    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    system_prompt = (
        "You analyze App Store review data for a product team. Identify 1-5 important "
        "recurring user problems. Use only the supplied evidence, do not invent facts, "
        "and make every recommendation specific and actionable. Write in English."
    )
    request_body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(build_context(analysis), ensure_ascii=False),
            },
        ],
        "response_format": _response_schema(),
    }

    for attempt in range(3):
        try:
            response = httpx.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=request_body,
                timeout=60.0,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            result = InsightsResult.model_validate_json(content)
            result.status = "generated"
            result.error = None
            return result
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 429 or status_code >= 500:
                if attempt < 2:
                    retry_after = exc.response.headers.get("retry-after", "")
                    delay = float(retry_after) if retry_after.isdigit() else 2**attempt
                    time.sleep(min(delay, 10.0))
                    continue
            detail = exc.response.text[:300].strip()
            raise InsightsError(f"Groq returned HTTP {status_code}: {detail}") from exc
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            raise InsightsError("Groq request failed after 3 attempts") from exc
        except (
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
            ValidationError,
        ) as exc:
            raise InsightsError("Groq returned a response that does not match the schema") from exc

    raise InsightsError("Groq request failed")
