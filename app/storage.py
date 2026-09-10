"""Store and resolve review collection files."""

import csv
from datetime import UTC, datetime
from pathlib import Path

from app.collector import CollectionResult


def save_collection(result: CollectionResult, data_root: Path) -> tuple[str, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    collection_id = f"{result.app.id}_{result.app.country}_{timestamp}"
    directory = data_root / collection_id
    directory.mkdir(parents=True, exist_ok=False)

    (directory / "reviews.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    rows = [review.model_dump() for review in result.reviews]
    with (directory / "reviews.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return collection_id, directory


def collection_directory(collection_id: str, data_root: Path) -> Path:
    directory = (data_root / collection_id).resolve()
    if data_root.resolve() not in directory.parents or not directory.is_dir():
        raise FileNotFoundError(collection_id)
    return directory
