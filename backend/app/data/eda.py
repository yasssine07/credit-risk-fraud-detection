from __future__ import annotations

import math
from typing import Any

import pandas as pd


def profile_dataframe(frame: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    """Return aggregate EDA diagnostics without returning source rows."""
    rows = len(frame)
    missing = frame.isna().sum()
    columns: list[dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        column: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "missing_count": int(missing[name]),
            "missing_percent": round(float(missing[name]) / rows * 100, 3) if rows else 0,
            "unique_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            values = series.dropna()
            if not values.empty:
                q = values.quantile([0.25, 0.5, 0.75])
                column["numeric_summary"] = {
                    "mean": _finite(values.mean()),
                    "std": _finite(values.std()),
                    "min": _finite(values.min()),
                    "q1": _finite(q.loc[0.25]),
                    "median": _finite(q.loc[0.5]),
                    "q3": _finite(q.loc[0.75]),
                    "max": _finite(values.max()),
                }
        else:
            column["top_categories"] = [
                {"value": str(value)[:120], "count": int(count)}
                for value, count in series.value_counts(dropna=True).head(10).items()
            ]
        columns.append(column)

    target_profile = None
    if target_column:
        if target_column not in frame.columns:
            raise ValueError(f"Target column '{target_column}' is not present in the uploaded CSV.")
        counts = frame[target_column].value_counts(dropna=False)
        target_profile = {
            "column": target_column,
            "class_counts": [{"label": str(label), "count": int(count)} for label, count in counts.items()],
            "minority_class_percent": round(float(counts.min()) / rows * 100, 4) if rows and len(counts) > 1 else None,
        }

    return {
        "row_count": rows,
        "column_count": len(frame.columns),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "total_missing_cells": int(missing.sum()),
        "columns": columns,
        "target": target_profile,
        "correlation": _correlation_summary(frame),
        "notice": "This is a descriptive profile of the uploaded data. It does not establish model quality or confirm fraud.",
    }


def _finite(value: Any) -> float | None:
    result = float(value)
    return round(result, 6) if math.isfinite(result) else None


def _correlation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"method": "pearson", "pairs": []}
    corr = numeric.corr(method="pearson")
    pairs = []
    names = list(corr.columns)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append({"feature_a": str(left), "feature_b": str(right), "correlation": round(float(value), 5)})
    pairs.sort(key=lambda pair: abs(pair["correlation"]), reverse=True)
    return {"method": "pearson", "pairs": pairs[:100]}
