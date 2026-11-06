import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.data.balancing import build_balanced_pipeline
from app.models.model_registry import list_models, register_pipeline, score_model


def test_registered_model_scores_with_thresholds_and_local_explanation(tmp_path):
    features = pd.DataFrame({"amount": list(range(40)), "channel": ["web", "shop"] * 20})
    labels = pd.Series([0] * 20 + [1] * 20)
    pipeline = build_balanced_pipeline(features, LogisticRegression(max_iter=1000))
    pipeline.fit(features, labels)
    registry = register_pipeline(
        pipeline, model_name="logistic_regression", task="fraud", feature_frame=features, positive_label=1, registry=tmp_path
    )
    result = score_model(
        registry["model_version"], {"amount": 25, "channel": "unknown"},
        task="fraud", low_threshold=0.3, high_threshold=0.7, registry=tmp_path,
    )
    assert result["task"] == "fraud"
    assert 0 <= result["probability"] <= 1
    assert result["classification"] in {"legitimate", "suspicious", "high-risk"}
    assert result["explanation"]["available"] is True
    assert list_models(registry=tmp_path)[0]["model_version"] == registry["model_version"]
