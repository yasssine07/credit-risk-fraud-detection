import sys
import types

import pandas as pd
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.database.models import AuditLog, Base, Dataset, MLModel, ModelMetric, TrainingExperiment, Transaction


def test_training_persistence_saves_metadata_but_not_raw_rows(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)
    session_module = types.ModuleType("app.database.session")
    session_module.SessionLocal = TestSession
    monkeypatch.setitem(sys.modules, "app.database.session", session_module)

    from app.services.persistence import persist_training_run

    frame = pd.DataFrame({"amount": [10, 20], "class": [0, 1]})
    result = {
        "partition_sizes": {"train": 1, "validation": 1, "test": 0},
        "split_strategy": "chronological",
        "validation": {"metrics": {"recall": 1.0, "pr_auc_average_precision": 1.0}},
    }
    saved = persist_training_run(
        filename="sample.csv", csv_bytes=b"amount,class\n10,0\n20,1\n", frame=frame,
        target_column="class", model_name="logistic_regression", balance_strategy="none",
        feature_selection="none", result=result,
    )
    assert saved["saved"] is True
    with TestSession() as session:
        assert session.scalar(select(func.count()).select_from(Dataset)) == 1
        assert session.scalar(select(func.count()).select_from(TrainingExperiment)) == 1
        assert session.scalar(select(func.count()).select_from(MLModel)) == 1
        assert session.scalar(select(func.count()).select_from(ModelMetric)) == 2
        assert session.scalar(select(func.count()).select_from(AuditLog)) == 1
        assert session.scalar(select(func.count()).select_from(Transaction)) == 0
        dataset = session.scalar(select(Dataset))
        assert dataset.content_sha256 and "columns" in dataset.schema_summary
