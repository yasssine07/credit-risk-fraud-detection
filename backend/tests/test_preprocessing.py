import numpy as np
import pandas as pd

from app.data.preprocessing import prepare_supervised_data
from app.services.training import train_and_evaluate
from app.services.evaluation import choose_threshold_for_fpr


def test_preprocessor_fits_on_train_and_handles_unseen_category():
    size = 40
    frame = pd.DataFrame({
        "amount": list(range(size - 2)) + [10000, 20000],
        "channel": ["web"] * (size - 2) + ["new-channel", "new-channel"],
        "label": [0, 1] * (size // 2),
    })
    split = prepare_supervised_data(frame, "label", test_fraction=0.2, validation_fraction=0.2)
    assert split.split_strategy == "stratified_random"
    assert split.X_train.shape[1] == len(split.feature_names)
    assert split.X_validation.shape[1] == split.X_test.shape[1]
    assert np.isfinite(split.X_test.toarray() if hasattr(split.X_test, "toarray") else split.X_test).all()


def test_time_split_preserves_order_and_excludes_timestamp():
    frame = pd.DataFrame({
        "event_time": pd.date_range("2025-01-01", periods=40, freq="h").astype(str),
        "amount": range(40),
        "label": [0, 1] * 20,
    })
    split = prepare_supervised_data(frame, "label", time_column="event_time")
    assert split.split_strategy == "chronological"
    assert split.y_train.index.max() < split.y_validation.index.min()
    assert split.y_validation.index.max() < split.y_test.index.min()
    assert all("event_time" not in feature for feature in split.feature_names)


def test_training_reports_validation_only_and_leaves_test_untouched():
    frame = pd.DataFrame({
        "amount": range(80),
        "label": [0] * 60 + [1] * 20,
    })
    result = train_and_evaluate(frame, target_column="label", model_name="logistic_regression")
    assert result["test_evaluated"] is False
    assert "recall" in result["validation"]["metrics"]
    assert "pr_auc_average_precision" in result["validation"]["metrics"]


def test_threshold_selection_obeys_false_positive_cap():
    labels = pd.Series([0] * 90 + [1] * 10)
    scores = np.linspace(0.0, 1.0, 100)
    selected = choose_threshold_for_fpr(labels, scores, max_false_positive_rate=0.05)
    assert selected["validation_false_positive_rate"] <= 0.05
    assert 0 <= selected["threshold"] <= 1
