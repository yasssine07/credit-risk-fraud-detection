import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from app.data.balancing import build_balanced_pipeline


def test_oversampling_pipeline_resamples_fit_rows_only():
    features = pd.DataFrame({"amount": range(40), "channel": ["web", "shop"] * 20})
    labels = pd.Series([0] * 28 + [1] * 4 + [0] * 8)
    pipeline = build_balanced_pipeline(features.iloc[:32], LogisticRegression(max_iter=1000), "random_over")
    pipeline.fit(features.iloc[:32], labels.iloc[:32])
    sampler = pipeline.named_steps["sampler"]
    assert len(sampler.sample_indices_) > 32
    assert len(features.iloc[32:]) == 8


def test_class_weight_strategy_sets_balanced_weight_without_sampler():
    features = pd.DataFrame({"amount": range(20)})
    pipeline = build_balanced_pipeline(features, LogisticRegression(), "class_weight")
    assert pipeline.named_steps["classifier"].class_weight == "balanced"
    assert pipeline.named_steps["sampler"] == "passthrough"


def test_feature_selector_is_fit_inside_training_pipeline():
    features = pd.DataFrame({"amount": range(40), "channel": ["web", "shop"] * 20})
    labels = pd.Series([0, 1] * 20)
    pipeline = build_balanced_pipeline(
        features, LogisticRegression(max_iter=1000), feature_selection="random_forest", selected_feature_count=1
    )
    pipeline.fit(features, labels)
    transformer = pipeline.named_steps["preprocessor"]
    assert sum(transformer.get_support()) == 1


def test_mutual_information_selection_clones_cleanly_inside_cross_validation():
    features = pd.DataFrame({"amount": range(40), "channel": ["web", "shop"] * 20})
    labels = pd.Series([0, 1] * 20)
    pipeline = build_balanced_pipeline(features, LogisticRegression(max_iter=1000), feature_selection="mutual_info", selected_feature_count=2)
    scores = cross_val_score(pipeline, features, labels, cv=StratifiedKFold(2, shuffle=True, random_state=4), scoring="average_precision")
    assert len(scores) == 2
