from __future__ import annotations

import csv
from io import BytesIO

import pandas as pd

MAX_DATASET_BYTES = 250 * 1024 * 1024


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset cannot be safely profiled."""


def read_csv_bytes(contents: bytes, filename: str | None) -> pd.DataFrame:
    if not filename or not filename.lower().endswith(".csv"):
        raise DatasetValidationError("Upload a .csv file.")
    if not contents:
        raise DatasetValidationError("The uploaded file is empty.")
    if len(contents) > MAX_DATASET_BYTES:
        raise DatasetValidationError("CSV exceeds the 250 MiB upload limit.")
    try:
        sample = contents[:65536].decode("utf-8-sig", errors="replace")
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
        frame = pd.read_csv(BytesIO(contents), sep=delimiter, encoding="utf-8-sig", low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetValidationError(f"Could not parse CSV: {exc}") from exc
    if frame.empty:
        raise DatasetValidationError("CSV contains headers but no data rows.")
    if not frame.columns.is_unique:
        raise DatasetValidationError("CSV has duplicate column names; rename them before profiling.")
    if len(frame.columns) > 500:
        raise DatasetValidationError("CSV exceeds the 500-column limit.")
    return frame
