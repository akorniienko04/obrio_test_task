import argparse
import asyncio
import sys
from pathlib import Path

from app.collector import CollectRequest, collect_reviews
from app.storage import save_collection


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Collect and clean App Store reviews.")
    result.add_argument("app", help="Numeric App Store ID or apps.apple.com URL")
    result.add_argument("--country", default="us")
    result.add_argument("--count", type=int, default=100)
    result.add_argument("--seed", type=int)
    result.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Base directory for collected review folders (default: data)",
    )
    return result


async def run(args: argparse.Namespace) -> int:
    try:
        result = await collect_reviews(
            CollectRequest(app=args.app, country=args.country, count=args.count, seed=args.seed)
        )
    except (ValueError, LookupError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _, run_directory = save_collection(result, args.output)
    print(f"Saved {result.collected_count} reviews to {run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parser().parse_args())))
