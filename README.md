# App Store Review Collection API

Small FastAPI service that collects, cleans, and analyzes Apple App Store
reviews.

## Current flow

```text
App ID or App Store URL
  → validate application through Apple's Lookup API
  → fetch reviews from available Apple storefronts
  → deduplicate and randomly sample up to 100 reviews
  → normalize text and remove URLs
  → save JSON and CSV under a collection ID
  → run sentiment, rating, and negative-term analysis
```

The collector uses Apple's public catalog reviews endpoint with pagination. It
throttles requests and, when necessary, combines the requested storefront with
`us`, `gb`, `ca`, and `au`, while preserving the source country on every review.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
uvicorn app.main:app --reload
```

To enable free local LLM-generated actionable insights, install Ollama and pull
a model before starting the server:

```bash
ollama pull llama3.2
export LLM_PROVIDER='ollama'  # default
export OLLAMA_MODEL='llama3.2'  # optional
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

## Collect reviews

```bash
curl -X POST http://127.0.0.1:8000/reviews/collect \
  -H 'Content-Type: application/json' \
  -d '{
    "app": "1459969523",
    "country": "gb",
    "count": 100,
    "seed": 42
  }'
```

`app` accepts a numeric App Store ID or an `apps.apple.com` URL. `country` is a
two-letter storefront code; `count` is from 1 to 100. `seed` is optional and
makes sampling reproducible.

Each returned review contains the original App Store fields and `cleaned_text`.
The clean text combines title and body, normalizes Unicode and whitespace, and
removes URLs. The response also contains `collection_id` and download URLs.

## Analyze through the API

Use the `collection_id` returned by `/reviews/collect`:

```bash
curl -X POST http://127.0.0.1:8000/reviews/analyze \
  -H 'Content-Type: application/json' \
  -d '{"collection_id":"1459969523_us_20260910T120000Z"}'
```

The endpoint creates `analysis.json` inside the collection directory and
returns sentiment distribution, rating metrics, negative keywords, and
two-word negative phrases. When Ollama is running, it also returns local
LLM-generated `actionable_insights` grounded in the negative terms and review
examples. As an alternative, set `LLM_PROVIDER=openai` and configure
`OPENAI_API_KEY`.

## Download raw reviews

```bash
curl -OJ \
  'http://127.0.0.1:8000/reviews/1459969523_us_20260910T120000Z/download?format=json'
```

Set `format=csv` to download the CSV version.

## Sample report

A sample report for Nebula is available at
[`reports/nebula_report.md`](reports/nebula_report.md).

## Command line

```bash
python scripts/collect_reviews.py 1459969523 --country gb --count 100 --seed 42 \
  --output data
```

Every run creates a separate directory, so previous collections are not
overwritten:

```text
data/
└── 1459969523_gb_20260910T120000Z/
    ├── reviews.json
    └── reviews.csv
```

## Analyze sentiment

```bash
python scripts/analyze_sentiment.py \
  --input data/1459969523_gb_20260910T120000Z/reviews.json
```

This reads `cleaned_text`, classifies every review as `positive`, `neutral`, or
`negative`, and writes `sentiment_reviews.json` beside the input. The output
also contains `sentiment_score`, `sentiment_distribution`, `rating_metrics`
(average and 1–5-star counts/percentages), and `negative_terms` (the most common
keywords and two-word phrases in negative reviews). The first run downloads the
multilingual model; later runs use the local model cache.

## Error handling

The endpoint returns standard HTTP errors:

```json
{ "detail": "Application 123 was not found in the us storefront" }
```

The client validates input, retries Apple HTTP 429/5xx errors, handles timeouts,
and reports partial collection when fewer reviews are available than requested.

## Tests

```bash
pytest -q
ruff check .
ruff format --check .
```
