from scripts.analyze_sentiment import add_sentiment, calculate_rating_metrics, normalize_label


def test_normalizes_model_labels() -> None:
    assert normalize_label("LABEL_0") == "negative"
    assert normalize_label("positive") == "positive"


def test_adds_sentiment_and_distribution_without_using_rating() -> None:
    payload = {
        "reviews": [
            {"id": "1", "rating": 5, "cleaned_text": "Text one"},
            {"id": "2", "rating": 1, "cleaned_text": "Text two"},
        ]
    }
    result = add_sentiment(
        payload,
        [{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.8}],
    )

    assert result["reviews"][0]["sentiment"] == "positive"
    assert result["reviews"][1]["sentiment_score"] == 0.8
    assert result["sentiment_distribution"]["positive"] == {"count": 1, "percentage": 50.0}
    assert result["rating_metrics"]["average_rating"] == 3.0
    assert result["negative_terms"]["keywords"][0]["term"] == "text"


def test_rating_distribution() -> None:
    result = calculate_rating_metrics([{"rating": 5}, {"rating": 5}, {"rating": 1}])
    assert result["average_rating"] == 3.67
    assert result["distribution"]["5"] == {"count": 2, "percentage": 66.67}
