"""Collect and clean a small random sample of App Store reviews."""

import asyncio
import random
import re
import unicodedata
from typing import Annotated, Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, StringConstraints, field_validator

APP_ID_PATTERN = re.compile(r"(?:^|/)id(?P<app_id>\d+)(?:[/?#]|$)")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
FALLBACK_STOREFRONTS = ("us", "gb", "ca", "au")
APPLE_REQUEST_DELAY_SECONDS = 3.0
CountryCode = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_lower=True, pattern=r"^[a-zA-Z]{2}$"),
]


class CollectRequest(BaseModel):
    app: str = Field(min_length=1, description="Numeric App Store ID or apps.apple.com URL")
    country: CountryCode = "us"
    count: int = Field(default=100, ge=1, le=100)
    seed: int | None = None

    @field_validator("app")
    @classmethod
    def strip_app(cls, value: str) -> str:
        return value.strip()


class AppInfo(BaseModel):
    id: str
    name: str
    country: str
    developer: str | None = None
    store_url: str | None = None


class Review(BaseModel):
    id: str
    title: str
    text: str
    rating: int = Field(ge=1, le=5)
    country: str
    author: str | None = None
    created_at: str | None = None
    cleaned_text: str


class CollectionResult(BaseModel):
    app: AppInfo
    requested_count: int
    collected_count: int
    available_pool_size: int
    is_partial: bool
    warnings: list[str] = Field(default_factory=list)
    reviews: list[Review]


def clean_text(title: str, text: str) -> str:
    combined = unicodedata.normalize("NFKC", f"{title}. {text}")
    return WHITESPACE_PATTERN.sub(" ", URL_PATTERN.sub(" ", combined)).strip()


def extract_app_id(value: str) -> str:
    if value.isdigit():
        return value
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname != "apps.apple.com":
        raise ValueError("app must be a numeric App Store ID or an apps.apple.com URL")
    match = APP_ID_PATTERN.search(parsed.path)
    if not match:
        raise ValueError("The App Store URL does not contain an app ID")
    return match.group("app_id")


async def collect_reviews(
    request: CollectRequest,
    http_client: httpx.AsyncClient | None = None,
) -> CollectionResult:
    """Fetch, deduplicate, randomly sample, and clean App Store reviews."""
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        follow_redirects=True,
        headers={"Accept": "application/json", "User-Agent": "app-store-review-analysis/0.1"},
    )
    try:
        app_id = extract_app_id(request.app)
        app = await _lookup_app(client, app_id, request.country)

        unique_reviews: dict[str, Review] = {}
        # The Catalog endpoint exposes paginated reviews. Requests are
        # deliberately throttled because Apple may otherwise return 429s.
        storefronts = dict.fromkeys((request.country, *FALLBACK_STOREFRONTS))
        used_storefronts: list[str] = []
        requested_pool_size = min(request.count + 20, 200)
        requested_rss = False
        for storefront in storefronts:
            for page in range(10):
                if requested_rss:
                    await asyncio.sleep(APPLE_REQUEST_DELAY_SECONDS)
                try:
                    reviews = await _reviews_page(client, app_id, storefront, page * 20)
                except RuntimeError:
                    reviews = []
                if not reviews and page == 0:
                    # Apple may temporarily return an empty catalog page while
                    # the legacy RSS feed is still available.
                    reviews = await _rss_reviews_page(client, app_id, storefront)
                requested_rss = True
                if not reviews:
                    break
                if storefront not in used_storefronts:
                    used_storefronts.append(storefront)
                unique_reviews.update({review.id: review for review in reviews})
                if len(unique_reviews) >= requested_pool_size:
                    break
            if len(unique_reviews) >= requested_pool_size:
                break

        pool = list(unique_reviews.values())
        if not pool:
            raise RuntimeError(
                "Apple returned no reviews for the available storefronts; try again later"
            )
        selected = random.Random(request.seed).sample(pool, k=min(request.count, len(pool)))
        is_partial = len(selected) < request.count
        warnings = ["Fewer reviews were available than requested"] if is_partial else []
        if len(used_storefronts) > 1:
            warnings.append(
                "Apple review results were combined from multiple storefronts: "
                f"{', '.join(used_storefronts)}"
            )
        return CollectionResult(
            app=app,
            requested_count=request.count,
            collected_count=len(selected),
            available_pool_size=len(pool),
            is_partial=is_partial,
            warnings=warnings,
            reviews=selected,
        )
    finally:
        if owns_client:
            await client.aclose()


async def _lookup_app(client: httpx.AsyncClient, app_id: str, country: str) -> AppInfo:
    payload = await _get_json(
        client,
        "https://itunes.apple.com/lookup",
        params={"id": app_id, "country": country, "entity": "software"},
    )
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Apple returned an unexpected lookup response")
    if not results:
        raise LookupError(f"Application {app_id} was not found in the {country} storefront")
    item = results[0]
    return AppInfo(
        id=app_id,
        name=str(item.get("trackName") or f"App {app_id}"),
        developer=item.get("artistName"),
        country=country,
        store_url=item.get("trackViewUrl"),
    )


async def _reviews_page(
    client: httpx.AsyncClient,
    app_id: str,
    country: str,
    offset: int,
) -> list[Review]:
    payload = await _get_json(
        client,
        f"https://apps.apple.com/api/apps/v1/catalog/{country}/apps/{app_id}/reviews",
        params={"platform": "web", "offset": str(offset), "limit": "20"},
        empty_on_404=True,
    )
    entries = payload.get("data", [])
    if not isinstance(entries, list):
        raise RuntimeError("Apple returned an unexpected reviews response")
    return [
        review for entry in entries if (review := _parse_catalog_review(entry, country)) is not None
    ]


async def _rss_reviews_page(client: httpx.AsyncClient, app_id: str, country: str) -> list[Review]:
    payload = await _get_json(
        client,
        f"https://itunes.apple.com/{country}/rss/customerreviews/"
        f"page=1/id={app_id}/sortby=mostrecent/json",
        params={},
        empty_on_404=True,
    )
    feed = payload.get("feed", {})
    entries = feed.get("entry", []) if isinstance(feed, dict) else []
    if not isinstance(entries, list):
        return []
    return [
        review for entry in entries if (review := _parse_rss_review(entry, country)) is not None
    ]


def _parse_catalog_review(entry: Any, country: str) -> Review | None:
    try:
        if not isinstance(entry, dict):
            return None
        attributes = entry["attributes"]
        title = str(attributes.get("title") or "")
        text = str(attributes.get("review") or "")
        return Review(
            id=str(entry["id"]),
            title=title,
            text=text,
            rating=int(attributes["rating"]),
            country=country,
            author=attributes.get("userName"),
            created_at=attributes.get("date"),
            cleaned_text=clean_text(title, text),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _parse_rss_review(entry: Any, country: str) -> Review | None:
    try:
        if not isinstance(entry, dict):
            return None
        title = str(entry["title"]["label"])
        text = str(entry["content"]["label"])
        return Review(
            id=str(entry["id"]["label"]),
            title=title,
            text=text,
            rating=int(entry["im:rating"]["label"]),
            country=country,
            author=entry.get("author", {}).get("name", {}).get("label"),
            created_at=entry.get("updated", {}).get("label"),
            cleaned_text=clean_text(title, text),
        )
    except (KeyError, TypeError, ValueError):
        return None


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, str],
    empty_on_404: bool = False,
) -> dict[str, Any]:
    for attempt in range(3):
        try:
            response = await client.get(url, params=params)
            if response.status_code == 404 and empty_on_404:
                return {"data": []}
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            raise RuntimeError("Apple returned invalid JSON")
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
            ValueError,
        ) as exc:
            retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                exc.response.status_code == 429 or exc.response.status_code >= 500
            )
            if not retryable or attempt == 2:
                raise RuntimeError("Apple review source is temporarily unavailable") from exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                retry_after = exc.response.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 10.0
                await asyncio.sleep(min(delay * (attempt + 1), 60.0))
            else:
                await asyncio.sleep(2**attempt)
    raise RuntimeError("Apple review source is temporarily unavailable")
