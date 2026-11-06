from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select

from app.database.models import AuditLog, Dataset, MLModel, ModelMetric, TrainingExperiment


def persist_training_run(
    *,
    filename: str,
    csv_bytes: bytes,
    frame: pd.DataFrame,
    target_column: str,
    model_name: str,
    balance_strategy: str,
    feature_selection: str,
    result: dict[str, object],
) -> dict[str, object]:
    """Persist experiment metadata/metrics, never raw uploaded feature rows."""
    from app.database.session import SessionLocal

    digest = hashlib.sha256(csv_bytes).hexdigest()
    with SessionLocal() as session, session.begin():
        dataset = session.scalar(select(Dataset).where(Dataset.content_sha256 == digest))
        if dataset is None:
            dataset = Dataset(
                name=filename[:200],
                source="user_upload",
                target_column=target_column,
                row_count=len(frame),
                schema_summary={"columns": [{"name": str(name), "dtype": str(dtype)} for name, dtype in frame.dtypes.items()]},
                content_sha256=digest,
            )
            session.add(dataset)
            session.flush()

        experiment_id = uuid.uuid4()
        experiment = TrainingExperiment(
            id=experiment_id,
            dataset_id=dataset.id,
            task="classification",
            status="completed",
            configuration={
                "model": model_name,
                "balance_strategy": balance_strategy,
                "feature_selection": feature_selection,
                "target_column": target_column,
                "partition_sizes": result["partition_sizes"],
            },
            split_strategy=result["split_strategy"],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        session.add(experiment)
        registry_record = result.get("registry") or {}
        model = MLModel(
            experiment_id=experiment_id,
            name=model_name,
            version=str(registry_record.get("model_version", experiment_id)),
            task="classification",
            artifact_uri=registry_record.get("artifact_uri"),
            is_active=False,
        )
        session.add(model)
        session.flush()
        metrics = result["validation"]["metrics"]
        for metric_name, metric_value in metrics.items():
            session.add(ModelMetric(model_id=model.id, partition="validation", metric_name=metric_name, metric_value=float(metric_value)))
        session.add(AuditLog(action="training.completed", entity_type="training_experiment", entity_id=str(experiment_id), details={"dataset_sha256": digest}))
        session.flush()
        return {
            "saved": True,
            "experiment_id": str(experiment_id),
            "model_version": model.version,
            "artifact_saved": False,
            "raw_rows_saved": False,
        }
