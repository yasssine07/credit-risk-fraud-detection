from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.balancing import BalanceStrategy, build_balanced_pipeline
from app.data.preprocessing import prepare_supervised_data
from app.models.machine_learning.classifiers import make_classifier
from app.services.evaluation import evaluate_classifier
from app.services.explainability import explain_classifier
from app.models.model_registry import register_pipeline


def train_and_evaluate(
    frame: pd.DataFrame,
    *,
    target_column: str,
    model_name: str,
    balance_strategy: BalanceStrategy = "none",
    time_column: str | None = None,
    random_state: int = 42,
    include_explanations: bool = False,
    feature_selection: str = "none",
    task: str = "fraud",
    save_model: bool = False,
    positive_label: str | None = None,
) -> dict[str, object]:
    if task not in {"fraud", "credit_risk"}:
        raise ValueError("Task must be 'fraud' or 'credit_risk'.")
    split = prepare_supervised_data(frame, target_column, time_column=time_column, random_state=random_state)
    training_features = frame.drop(columns=[target_column] + ([time_column] if time_column else []))
    model = build_balanced_pipeline(
        training_features,
        make_classifier(model_name, random_state),
        balance_strategy,
        random_state=random_state,
        feature_selection=feature_selection,
    )
    model.fit(
        frame.loc[split.y_train.index].drop(columns=[target_column] + ([time_column] if time_column else [])),
        split.y_train,
    )
    classes = np.unique(split.y_train).tolist()
    resolved_positive = classes[-1] if positive_label is None else next((value for value in classes if str(value) == positive_label), None)
    if resolved_positive is None:
        raise ValueError("positive_label must match one of the target's observed class labels.")
    report = evaluate_classifier(
        model,
        frame.loc[split.y_validation.index].drop(columns=[target_column] + ([time_column] if time_column else [])),
        split.y_validation,
        positive_label=resolved_positive,
    )
    result = {
        "model": model_name,
        "balance_strategy": balance_strategy,
        "feature_selection": feature_selection,
        "split_strategy": split.split_strategy,
        "partition_sizes": {
            "train": len(split.y_train),
            "validation": len(split.y_validation),
            "test": len(split.y_test),
        },
        "validation": report,
        "test_evaluated": False,
        "notice": "Validation metrics use the validation partition only. The held-out test partition has not been evaluated.",
    }
    if include_explanations:
        train_rows = frame.loc[split.y_train.index].drop(columns=[target_column] + ([time_column] if time_column else []))
        validation_rows = frame.loc[split.y_validation.index].drop(columns=[target_column] + ([time_column] if time_column else []))
        result["explanations"] = explain_classifier(model, train_rows, validation_rows)
    if save_model:
        train_rows = frame.loc[split.y_train.index].drop(columns=[target_column] + ([time_column] if time_column else []))
        result["registry"] = register_pipeline(
            model,
            model_name=model_name,
            task=task,
            feature_frame=train_rows,
            positive_label=resolved_positive,
        )
    return result
