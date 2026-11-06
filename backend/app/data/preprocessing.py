from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class SupervisedDataSplit:
    """Leakage-safe prepared partitions and the train-fitted transformer."""

    X_train: object
    X_validation: object
    X_test: object
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: list[str]
    split_strategy: str


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [name for name in features.columns if name not in numeric_columns]

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_supervised_data(
    frame: pd.DataFrame,
    target_column: str,
    *,
    time_column: str | None = None,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
    random_state: int = 42,
) -> SupervisedDataSplit:
    """Split first, then fit preprocessing only on training rows.

    If `time_column` is set, rows are sorted by that timestamp and split
    chronologically. The timestamp is excluded from model features by default.
    Otherwise, stratified two-stage random splitting is used.
    """
    if target_column not in frame:
        raise ValueError(f"Target column '{target_column}' is not present.")
    if frame[target_column].isna().any():
        raise ValueError("Target column contains missing labels; resolve them before training.")
    if not 0 < test_fraction < 0.5 or not 0 < validation_fraction < 0.5 or test_fraction + validation_fraction >= 0.8:
        raise ValueError("Test and validation fractions must be positive and sum to less than 0.8.")
    if len(frame) < 10:
        raise ValueError("At least 10 rows are required for train/validation/test splitting.")

    ordered = frame.copy()
    if time_column:
        if time_column not in ordered:
            raise ValueError(f"Time column '{time_column}' is not present.")
        timestamps = pd.to_datetime(ordered[time_column], errors="coerce", utc=True)
        if timestamps.isna().any():
            raise ValueError("Time column has missing or invalid timestamps.")
        ordered = ordered.loc[timestamps.sort_values(kind="stable").index]
        n_test = max(1, round(len(ordered) * test_fraction))
        n_validation = max(1, round(len(ordered) * validation_fraction))
        train_end = len(ordered) - n_validation - n_test
        if train_end < 2:
            raise ValueError("Not enough rows remain in the training partition for these split fractions.")
        train = ordered.iloc[:train_end]
        validation = ordered.iloc[train_end : len(ordered) - n_test]
        test = ordered.iloc[len(ordered) - n_test :]
        strategy = "chronological"
    else:
        from sklearn.model_selection import train_test_split

        stratify = _valid_stratification(ordered[target_column])
        try:
            train_validation, test = train_test_split(
                ordered,
                test_size=test_fraction,
                random_state=random_state,
                stratify=stratify,
            )
            first_split_stratified = stratify is not None
        except ValueError:
            train_validation, test = train_test_split(
                ordered, test_size=test_fraction, random_state=random_state, stratify=None
            )
            first_split_stratified = False
        relative_validation = validation_fraction / (1 - test_fraction)
        stratify_train = _valid_stratification(train_validation[target_column])
        try:
            train, validation = train_test_split(
                train_validation,
                test_size=relative_validation,
                random_state=random_state,
                stratify=stratify_train,
            )
            second_split_stratified = stratify_train is not None
        except ValueError:
            train, validation = train_test_split(
                train_validation, test_size=relative_validation, random_state=random_state, stratify=None
            )
            second_split_stratified = False
        strategy = "stratified_random" if first_split_stratified and second_split_stratified else "random_unstratified"

    excluded = {target_column}
    if time_column:
        excluded.add(time_column)
    columns = [name for name in frame.columns if name not in excluded]
    if not columns:
        raise ValueError("No feature columns remain after excluding the target and time column.")

    X_train_raw = train[columns]
    X_validation_raw = validation[columns]
    X_test_raw = test[columns]
    y_train, y_validation, y_test = train[target_column], validation[target_column], test[target_column]

    preprocessor = make_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_validation = preprocessor.transform(X_validation_raw)
    X_test = preprocessor.transform(X_test_raw)
    names = [str(name) for name in preprocessor.get_feature_names_out()]

    return SupervisedDataSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=names,
        split_strategy=strategy,
    )


def _valid_stratification(target: pd.Series) -> pd.Series | None:
    counts = target.value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    return target
