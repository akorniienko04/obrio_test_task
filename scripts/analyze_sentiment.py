"""Classify collected reviews as positive, neutral, or negative."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
LABELS = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}
STOP_WORDS = {
    # Common English function words that do not describe a product problem.
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "with",
    "about",
    "as",
    "into",
    "after",
    "before",
    "than",
    "through",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "it's",
    "i",
    "i'm",
    "im",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "you",
    "your",
    "yours",
    "they",
    "their",
    "theirs",
    "them",
    "he",
    "his",
    "she",
    "her",
    "hers",
    "was",
    "were",
    "is",
    "are",
    "be",
    "been",
    "being",
    "am",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "would",
    "could",
    "should",
    "will",
    "can",
    "not",
    "don't",
    "don’t",
    "doesn't",
    "didn't",
    "but",
    "if",
    "then",
    "when",
    "what",
    "which",
    "who",
    "how",
    "why",
    "all",
    "any",
    "some",
    "more",
    "most",
    "very",
    "just",
    "even",
    "also",
    "only",
    "out",
    "up",
    "down",
    "get",
    "got",
    "like",
    "feel",
    "felt",
    "because",
    "app",
    "one",
    "two",
    "really",
    "что",
    "это",
    "как",
    "та",
    "і",
    "й",
    "але",
    "не",
    "дуже",
    "це",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Analyze review sentiment from cleaned text.")
    result.add_argument("--input", type=Path, required=True, help="Path to reviews.json")
    result.add_argument(
        "--output", type=Path, help="Output JSON path (default: sentiment_reviews.json)"
    )
    result.add_argument("--model", default=DEFAULT_MODEL)
    result.add_argument("--batch-size", type=int, default=16)
    return result


def load_input(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Input file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), list):
        raise ValueError("Input JSON must contain a reviews list")
    return payload


def normalize_label(label: str) -> str:
    normalized = LABELS.get(label.lower())
    if normalized is None:
        raise ValueError(f"Unsupported sentiment label returned by model: {label}")
    return normalized


def add_sentiment(payload: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    reviews = [review for review in payload["reviews"] if isinstance(review, dict)]
    if len(reviews) != len(predictions):
        raise ValueError("The model returned a different number of predictions than reviews")
    labels: list[str] = []
    for review, prediction in zip(reviews, predictions, strict=True):
        label = normalize_label(str(prediction["label"]))
        review["sentiment"] = label
        review["sentiment_score"] = round(float(prediction["score"]), 4)
        labels.append(label)
    counts = Counter(labels)
    total = len(labels)
    payload["sentiment_distribution"] = {
        label: {
            "count": counts[label],
            "percentage": round(counts[label] / total * 100, 2) if total else 0.0,
        }
        for label in ("positive", "neutral", "negative")
    }
    payload["rating_metrics"] = calculate_rating_metrics(reviews)
    payload["negative_terms"] = find_negative_terms(reviews)
    return payload


def calculate_rating_metrics(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    ratings = [int(review["rating"]) for review in reviews if review.get("rating") is not None]
    counts = {str(stars): sum(rating == stars for rating in ratings) for stars in range(1, 6)}
    total = len(ratings)
    return {
        "average_rating": round(sum(ratings) / total, 2) if total else None,
        "total_reviews": total,
        "distribution": {
            stars: {"count": count, "percentage": round(count / total * 100, 2) if total else 0.0}
            for stars, count in counts.items()
        },
    }


def find_negative_terms(
    reviews: list[dict[str, Any]], limit: int = 20
) -> dict[str, list[dict[str, Any]]]:
    negative = [review for review in reviews if review.get("sentiment") == "negative"]
    token_lists = []
    for review in negative:
        tokens = [
            token
            for token in re.findall(r"[\w’']+", str(review.get("cleaned_text", "")).lower())
            if len(token) > 2 and token not in STOP_WORDS and not token.isdigit()
        ]
        token_lists.append(tokens)
    words = Counter(token for tokens in token_lists for token in tokens)
    phrases = Counter(
        f"{tokens[index]} {tokens[index + 1]}"
        for tokens in token_lists
        for index in range(len(tokens) - 1)
    )
    return {
        "keywords": [{"term": term, "count": count} for term, count in words.most_common(limit)],
        "phrases": [
            {"phrase": phrase, "count": count} for phrase, count in phrases.most_common(limit)
        ],
    }


def run(args: argparse.Namespace) -> None:
    from transformers import pipeline

    payload = load_input(args.input)
    reviews = [review for review in payload["reviews"] if isinstance(review, dict)]
    texts = [str(review.get("cleaned_text", "")) for review in reviews]
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Every review must contain a non-empty cleaned_text")

    output = args.output or args.input.with_name("sentiment_reviews.json")
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose another --output path.")

    model = pipeline("sentiment-analysis", model=args.model)
    predictions = model(texts, batch_size=args.batch_size, truncation=True)
    result = add_sentiment(payload, predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved sentiment analysis for {len(reviews)} reviews to {output}")


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
