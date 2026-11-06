from __future__ import annotations

import numpy as np
import pandas as pd


def explain_classifier(
    fitted_pipeline,
    background: pd.DataFrame,
    instances: pd.DataFrame,
    *,
    max_background_rows: int = 20,
    max_explanation_rows: int = 20,
    max_features: int = 60,
    top_features: int = 8,
    random_state: int = 42,
) -> dict[str, object]:
    """Explain predictions in probability space for a fitted pipeline.

    Inputs should be representative, already-approved rows and must follow the
    same raw feature schema used when fitting. SHAP sees only transformed
    feature values and never receives the target column.
    """
    if not hasattr(fitted_pipeline, "named_steps") or not {"preprocessor", "classifier"}.issubset(fitted_pipeline.named_steps):
        raise ValueError("Expected a fitted preprocessing/classifier pipeline.")
    if background.empty or instances.empty:
        raise ValueError("Background and explained rows must not be empty.")
    if max_background_rows < 1 or max_explanation_rows < 1 or top_features < 1:
        raise ValueError("SHAP row and feature limits must be positive.")
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP support is optional. Install backend/requirements-xai.txt.") from exc

    background = background.sample(min(len(background), max_background_rows), random_state=random_state)
    instances = instances.sample(min(len(instances), max_explanation_rows), random_state=random_state)
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    names = [str(item) for item in preprocessor.get_feature_names_out()]
    if len(names) > max_features:
        raise ValueError(f"Transformed feature count ({len(names)}) exceeds the SHAP limit ({max_features}). Select or group features first.")
    background_values = _dense(preprocessor.transform(background))
    instance_values = _dense(preprocessor.transform(instances))
    classes = list(classifier.classes_)
    positive_index = len(classes) - 1

    explainer = shap.Explainer(classifier.predict_proba, background_values, algorithm="permutation")
    explanation = explainer(instance_values, max_evals=2 * len(names) + 1)
    values = np.asarray(explanation.values)
    bases = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, positive_index]
    if bases.ndim > 1:
        bases = bases[:, positive_index]
    means = np.mean(np.abs(values), axis=0)
    order = np.argsort(means)[::-1]
    global_importance = [
        {"feature": names[int(index)], "mean_absolute_shap": float(means[index])}
        for index in order[: min(top_features, len(order))]
    ]
    local = []
    for row_index, row_values in enumerate(values):
        top = np.argsort(np.abs(row_values))[::-1][: min(top_features, len(names))]
        local.append({
            "row_index": str(instances.index[row_index]),
            "base_value": float(np.asarray(bases).reshape(-1)[row_index]),
            "predicted_probability": float(classifier.predict_proba(instance_values[row_index : row_index + 1])[0, positive_index]),
            "top_contributions": [
                {"feature": names[int(index)], "value": float(instance_values[row_index, index]), "shap_value": float(row_values[index])}
                for index in top
            ],
        })
    return {
        "positive_label": _json_scalar(classes[positive_index]),
        "explained_row_count": len(instances),
        "background_row_count": len(background),
        "global_importance": global_importance,
        "local_explanations": local,
        "notice": "SHAP contributions describe this fitted model's behavior; they are not causal effects or financial decisions.",
    }


def _dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def _json_scalar(value):
    return value.item() if hasattr(value, "item") else value
