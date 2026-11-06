from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.data.eda import profile_dataframe
from app.data import ingestion
from app.data.ingestion import DatasetValidationError, read_csv_bytes


def test_csv_profile_reports_real_summary():
    frame = read_csv_bytes(b"amount,class\n10,0\n100,1\n", "transactions.csv")
    profile = profile_dataframe(frame, "class")
    assert profile["row_count"] == 2
    assert profile["target"]["class_counts"] == [{"label": "0", "count": 1}, {"label": "1", "count": 1}]


def test_invalid_extension_is_rejected():
    try:
        read_csv_bytes(b"a\n1\n", "transactions.xlsx")
    except DatasetValidationError as error:
        assert ".csv" in str(error)
    else:
        raise AssertionError("Expected invalid extension to fail")


def test_semicolon_csv_is_detected():
    frame = read_csv_bytes("amount;Class\n10;0\n100;1\n".encode(), "french-export.csv")
    assert list(frame.columns) == ["amount", "Class"]
    assert len(frame) == 2


def test_upload_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(ingestion, "MAX_DATASET_BYTES", 8)
    with pytest.raises(DatasetValidationError, match="upload limit"):
        read_csv_bytes(b"amount,class\n10,0\n", "large.csv")


def test_profile_endpoint_accepts_csv_upload():
    client = TestClient(app)
    response = client.post(
        "/api/v1/datasets/profile",
        files={"file": ("sample.csv", "amount,class\n10,0\n100,1\n", "text/csv")},
        data={"target_column": "class"},
    )
    assert response.status_code == 200
    assert response.json()["profile"]["row_count"] == 2
