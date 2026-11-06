from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from app.data.balancing import BalanceStrategy, build_balanced_pipeline
from app.models.machine_learning.classifiers import make_classifier


def optimize_random_forest(
    training_frame: pd.DataFrame,
    *,
    target_column: str,
    trials: int = 20,
    balance_strategy: BalanceStrategy = "none",
    random_state: int = 42,
    artifact_path: str | None = None,
) -> dict[str, object]:
    """Optimize PR-AUC by stratified CV on a training partition only.

    The caller is responsible for withholding validation/test data before
    invoking this function. Each CV fold clones the preprocessing and sampler
    pipeline, so resampling only touches the fold's training rows.
    """
    if target_column not in training_frame:
        raise ValueError(f"Target column '{target_column}' is not present.")
    if trials < 1 or trials > 500:
        raise ValueError("Trial count must be between 1 and 500.")
    target = training_frame[target_column]
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Optimization currently requires a complete binary target.")
    min_class = int(target.value_counts().min())
    folds = min(5, min_class)
    if folds < 2:
        raise ValueError("Each target class needs at least two samples for cross-validation.")

    import optuna

    features = training_frame.drop(columns=[target_column])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        estimator = make_classifier("random_forest", random_state).set_params(**params)
        pipeline = build_balanced_pipeline(features, estimator, balance_strategy, random_state=random_state)
        scores = cross_val_score(pipeline, features, target, cv=splitter, scoring="average_precision", n_jobs=1)
        return float(scores.mean())

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params = dict(study.best_params)
    best_model = make_classifier("random_forest", random_state).set_params(**best_params)
    fitted = build_balanced_pipeline(features, best_model, balance_strategy, random_state=random_state)
    fitted.fit(features, target)
    saved_to = None
    if artifact_path:
        from joblib import dump

        destination = Path(artifact_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        dump(fitted, destination)
        saved_to = os.fspath(destination)
    return {
        "objective": "mean_average_precision",
        "best_validation_score": float(study.best_value),
        "best_parameters": best_params,
        "trial_count": len(study.trials),
        "optimization_history": [
            {"trial": trial.number, "score": float(trial.value) if trial.value is not None else None}
            for trial in study.trials
        ],
        "artifact_path": saved_to,
        "notice": "The best model was refit on the supplied training partition. Keep all validation and test data out of this function.",
    }
