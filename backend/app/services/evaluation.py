from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(estimator, X, y_true: pd.Series, *, threshold: float = 0.5, positive_label=None) -> dict[str, object]:
    """Calculate thresholded binary metrics with recall/PR-AUC foregrounded."""
    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError("The initial evaluation service supports binary targets only.")
    positive_label = _resolve_positive_label(classes, positive_label)
    probabilities = _positive_scores(estimator, X, positive_label)
    predictions = np.where(probabilities >= threshold, positive_label, classes[0])
    matrix = confusion_matrix(y_true, predictions, labels=classes)
    tn, fp, fn, tp = matrix.ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0

    return {
        "positive_label": _json_scalar(positive_label),
        "threshold": threshold,
        "metrics": {
            "recall": float(recall_score(y_true, predictions, pos_label=positive_label, zero_division=0)),
            "pr_auc_average_precision": float(average_precision_score(y_true == positive_label, probabilities)),
            "precision": float(precision_score(y_true, predictions, pos_label=positive_label, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, pos_label=positive_label, zero_division=0)),
            "specificity": specificity,
            "roc_auc": float(roc_auc_score(y_true == positive_label, probabilities)),
            "matthews_correlation": float(matthews_corrcoef(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "accuracy": float(accuracy_score(y_true, predictions)),
        },
        "confusion_matrix": {"labels": [_json_scalar(c) for c in classes], "values": matrix.tolist()},
    }


def _positive_scores(estimator, X, positive_label) -> np.ndarray:
    classes = list(estimator.classes_)
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        return probabilities[:, classes.index(positive_label)]
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        if scores.ndim == 2:
            scores = scores[:, classes.index(positive_label)]
        # Map margins to a bounded score suitable for thresholding and ranking.
        return 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
    raise ValueError("Estimator must provide predict_proba or decision_function.")


def choose_threshold_for_fpr(y_true: pd.Series, positive_scores: np.ndarray, *, max_false_positive_rate: float = 0.01) -> dict[str, float]:
    """Choose the threshold with highest recall under an explicit FPR cap."""
    if not 0 < max_false_positive_rate < 1:
        raise ValueError("max_false_positive_rate must be between 0 and 1.")
    classes = np.unique(y_true)
    if len(classes) != 2 or len(positive_scores) != len(y_true):
        raise ValueError("Threshold selection requires binary labels and one score per row.")
    fpr, recall, thresholds = roc_curve(y_true == classes[-1], positive_scores)
    eligible = np.flatnonzero(fpr <= max_false_positive_rate)
    # The final ROC threshold can be infinity and always has zero FPR. Prefer
    # finite thresholds where available; otherwise keep the conservative fallback.
    finite = eligible[np.isfinite(thresholds[eligible])]
    candidates = finite if len(finite) else eligible
    best = int(candidates[np.argmax(recall[candidates])])
    return {
        "threshold": float(thresholds[best]),
        "validation_recall": float(recall[best]),
        "validation_false_positive_rate": float(fpr[best]),
        "max_false_positive_rate": max_false_positive_rate,
    }


def _resolve_positive_label(classes, requested):
    if requested is None:
        return classes[-1]
    for label in classes:
        if label == requested or str(label) == str(requested):
            return label
    raise ValueError(f"Positive label '{requested}' is not present in the target classes.")


def _json_scalar(value):
    return value.item() if hasattr(value, "item") else value
