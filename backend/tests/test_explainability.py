import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

pytest.importorskip("shap")

from app.data.balancing import build_balanced_pipeline
from app.services.explainability import explain_classifier


def test_shap_returns_global_and_row_level_probability_contributions():
    frame = pd.DataFrame({"amount": range(40), "channel": ["web", "shop"] * 20})
    labels = pd.Series([0] * 20 + [1] * 20)
    pipeline = build_balanced_pipeline(frame, LogisticRegression(max_iter=1000))
    pipeline.fit(frame, labels)
    result = explain_classifier(pipeline, frame.iloc[:20], frame.iloc[20:22], max_features=10)
    assert result["explained_row_count"] == 2
    assert result["global_importance"]
    assert result["local_explanations"][0]["top_contributions"]
