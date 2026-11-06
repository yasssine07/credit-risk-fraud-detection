from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "ml_artifacts" / "trained_models"


def register_pipeline(pipeline, *, model_name: str, task: str, feature_frame: pd.DataFrame, positive_label=None, registry: Path | None = None) -> dict[str, object]:
    root = (registry or DEFAULT_REGISTRY).resolve()
    root.mkdir(parents=True, exist_ok=True)
    version = str(uuid.uuid4())
    transformer = pipeline.named_steps["preprocessor"]
    transformed = transformer.transform(feature_frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    background = np.asarray(transformed, dtype=float).mean(axis=0, keepdims=True)
    metadata = {
        "model_version": version,
        "model_name": model_name,
        "task": task,
        "feature_names": [str(name) for name in feature_frame.columns],
        "positive_label": _scalar(pipeline.named_steps["classifier"].classes_[-1] if positive_label is None else positive_label),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact_tmp = root / f".{version}.joblib.tmp"
    metadata_tmp = root / f".{version}.json.tmp"
    joblib.dump({"pipeline": pipeline, "background": background, "metadata": metadata}, artifact_tmp)
    metadata_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    artifact_tmp.replace(root / f"{version}.joblib")
    metadata_tmp.replace(root / f"{version}.json")
    return metadata | {"artifact_saved": True, "artifact_uri": f"trained_models/{version}.joblib"}


def list_models(*, registry: Path | None = None) -> list[dict[str, object]]:
    root = (registry or DEFAULT_REGISTRY).resolve()
    if not root.exists():
        return []
    records = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("*.json")]
    return sorted(records, key=lambda item: item["created_at"], reverse=True)


def load_model(version: str, *, registry: Path | None = None) -> dict[str, object]:
    try:
        version = str(uuid.UUID(version))
    except ValueError as exc:
        raise ValueError("Invalid model version identifier.") from exc
    root = (registry or DEFAULT_REGISTRY).resolve()
    metadata_path = (root / f"{version}.json").resolve()
    artifact_path = (root / f"{version}.joblib").resolve()
    if metadata_path.parent != root or artifact_path.parent != root or not metadata_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError("Model version was not found in the local registry.")
    bundle = joblib.load(artifact_path)
    return bundle


def score_model(version: str, features: dict[str, object], *, task: str, low_threshold: float, high_threshold: float, registry: Path | None = None) -> dict[str, object]:
    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= low_threshold < high_threshold <= 1.")
    bundle = load_model(version, registry=registry)
    metadata = bundle["metadata"]
    if metadata["task"] != task:
        raise ValueError(f"This model is registered for task '{metadata['task']}', not '{task}'.")
    expected = metadata["feature_names"]
    unknown = set(features) - set(expected)
    if unknown:
        raise ValueError(f"Unknown input feature(s): {', '.join(sorted(unknown))}")
    row = pd.DataFrame([{name: features.get(name) for name in expected}], columns=expected)
    pipeline = bundle["pipeline"]
    transformed = pipeline.named_steps["preprocessor"].transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    positive_index = classes.index(metadata["positive_label"])
    probability = float(classifier.predict_proba(transformed)[0, positive_index])
    if probability < low_threshold:
        classification = "legitimate" if task == "fraud" else "low-risk"
    elif probability < high_threshold:
        classification = "suspicious" if task == "fraud" else "medium-risk"
    else:
        classification = "high-risk"
    explanation = _explain(classifier, bundle["background"], transformed, pipeline.named_steps["preprocessor"].get_feature_names_out(), positive_index)
    response: dict[str, object] = {
        "model_version": version,
        "task": task,
        "probability": probability,
        "classification": classification,
        "thresholds": {"low": low_threshold, "high": high_threshold},
        "explanation": explanation,
        "notice": "This model estimate supports human review; it is not a confirmed fraud finding or financial decision.",
    }
    if task == "credit_risk":
        response["credit_risk_score"] = round(probability * 1000, 1)
        response["score_definition"] = "Default probability multiplied by 1,000; higher values indicate higher model-estimated risk."
    return response


def _explain(classifier, background, instance, feature_names, positive_index: int) -> dict[str, object]:
    if len(feature_names) > 60:
        return {"available": False, "reason": "SHAP inference is capped at 60 transformed features."}
    try:
        import shap
    except ImportError:
        return {"available": False, "reason": "Install requirements-xai.txt to enable SHAP prediction explanations."}
    explainer = shap.Explainer(classifier.predict_proba, background, algorithm="permutation")
    explanation = explainer(instance, max_evals=2 * len(feature_names) + 1)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, positive_index]
    row_values = values[0]
    order = np.argsort(np.abs(row_values))[::-1][:8]
    return {
        "available": True,
        "baseline": "mean transformed training vector",
        "top_contributions": [
            {"feature": str(feature_names[index]), "shap_value": float(row_values[index])}
            for index in order
        ],
    }


def _scalar(value):
    return value.item() if hasattr(value, "item") else value
