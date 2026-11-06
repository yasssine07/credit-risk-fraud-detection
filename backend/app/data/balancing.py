from __future__ import annotations

from functools import partial
from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from app.data.preprocessing import make_preprocessor

BalanceStrategy = Literal["none", "random_over", "random_under", "smote", "adasyn", "smoteenn", "class_weight"]


class LeakageSafeFeatureTransformer(BaseEstimator, TransformerMixin):
    """Fit preprocessing and optional selection together within each CV fold."""

    def __init__(self, feature_selection: str = "none", selected_feature_count: int = 20, random_state: int = 42):
        self.feature_selection = feature_selection
        self.selected_feature_count = selected_feature_count
        self.random_state = random_state

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Feature selection requires training labels.")
        if self.feature_selection not in {"none", "mutual_info", "rfe", "random_forest"}:
            raise ValueError("Feature selection must be none, mutual_info, rfe, or random_forest.")
        self.preprocessor_ = make_preprocessor(X)
        transformed = self.preprocessor_.fit_transform(X, y)
        self.feature_names_in_ = [str(name) for name in self.preprocessor_.get_feature_names_out()]
        count = min(self.selected_feature_count, transformed.shape[1])
        self.selector_ = _make_selector(self.feature_selection, count, self.preprocessor_, self.random_state)
        if self.selector_ != "passthrough":
            self.selector_.fit(transformed, y)
        return self

    def transform(self, X):
        transformed = self.preprocessor_.transform(X)
        return transformed if self.selector_ == "passthrough" else self.selector_.transform(transformed)

    def get_feature_names_out(self, input_features=None):
        names = self.feature_names_in_
        if self.selector_ != "passthrough" and hasattr(self.selector_, "get_support"):
            names = [name for name, keep in zip(names, self.selector_.get_support()) if keep]
        return names

    def get_support(self):
        if self.selector_ == "passthrough":
            return [True] * len(self.feature_names_in_)
        return self.selector_.get_support()


def build_balanced_pipeline(
    features: pd.DataFrame,
    estimator,
    strategy: BalanceStrategy = "none",
    *,
    random_state: int = 42,
    feature_selection: str = "none",
    selected_feature_count: int = 20,
):
    """Build a fold-safe preprocessing, resampling, estimator pipeline.

    Use this complete pipeline inside cross-validation. imbalanced-learn's
    pipeline calls `fit_resample` only during `fit`, so validation/test rows
    are transformed and scored at their original class distribution.
    """
    valid = {"none", "random_over", "random_under", "smote", "adasyn", "smoteenn", "class_weight"}
    if strategy not in valid:
        raise ValueError(f"Unknown balancing strategy: {strategy}")
    if feature_selection not in {"none", "mutual_info", "rfe", "random_forest"}:
        raise ValueError("Feature selection must be none, mutual_info, rfe, or random_forest.")
    if selected_feature_count < 1:
        raise ValueError("selected_feature_count must be positive.")
    if features.empty or features.shape[1] == 0:
        raise ValueError("At least one training feature is required.")
    classifier = clone(estimator)
    if strategy == "class_weight":
        params = classifier.get_params(deep=False)
        if "class_weight" not in params:
            raise ValueError("This estimator does not expose class_weight; use an estimator-specific weighting option.")
        classifier.set_params(class_weight="balanced")
        sampler = "passthrough"
    else:
        sampler = _make_sampler(strategy, random_state)

    # Import lazily so ingestion and EDA remain usable without optional
    # resampling extras installed in minimal environments.
    from imblearn.pipeline import Pipeline

    return Pipeline([
        ("preprocessor", LeakageSafeFeatureTransformer(feature_selection, selected_feature_count, random_state)),
        ("sampler", sampler),
        ("classifier", classifier),
    ])


def _make_sampler(strategy: BalanceStrategy, random_state: int):
    if strategy in {"none", "class_weight"}:
        return "passthrough"
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    if strategy == "random_over":
        return RandomOverSampler(random_state=random_state)
    if strategy == "random_under":
        return RandomUnderSampler(random_state=random_state)
    if strategy == "smote":
        return SMOTE(random_state=random_state)
    if strategy == "adasyn":
        return ADASYN(random_state=random_state)
    if strategy == "smoteenn":
        return SMOTEENN(random_state=random_state)
    raise ValueError(f"Unknown balancing strategy: {strategy}")


def _make_selector(strategy: str, count: int, preprocessor: ColumnTransformer, random_state: int):
    if strategy == "mutual_info":
        return SelectKBest(score_func=partial(_mutual_information_scores, preprocessor=preprocessor, random_state=random_state), k=count)
    if strategy == "rfe":
        return RFE(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state), n_features_to_select=count)
    if strategy == "random_forest":
        return SelectFromModel(
            RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=random_state),
            threshold=-float("inf"),
            max_features=count,
        )
    return "passthrough"


def _mutual_information_scores(X, y, *, preprocessor, random_state: int):
    # Numeric dimensions remain continuous; one-hot categorical dimensions are discrete.
    mask = [False] * X.shape[1]
    categorical_slice = preprocessor.output_indices_.get("categorical", slice(0, 0))
    for index in range(categorical_slice.start or 0, categorical_slice.stop or 0):
        mask[index] = True
    return mutual_info_classif(X, y, discrete_features=mask, random_state=random_state)
