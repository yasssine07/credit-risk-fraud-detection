from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="analyst")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    experiments: Mapped[list[TrainingExperiment]] = relationship(back_populates="user")


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(100), default="user_upload")
    target_column: Mapped[str | None] = mapped_column(String(128), nullable=True)
    time_column: Mapped[str | None] = mapped_column(String(128), nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    schema_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    experiments: Mapped[list[TrainingExperiment]] = relationship(back_populates="dataset")


class TrainingExperiment(Base):
    __tablename__ = "training_experiments"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    task: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(24), default="completed")
    configuration: Mapped[dict] = mapped_column(JSON, default=dict)
    split_strategy: Mapped[str | None] = mapped_column(String(40), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dataset: Mapped[Dataset | None] = relationship(back_populates="experiments")
    user: Mapped[User | None] = relationship(back_populates="experiments")
    models: Mapped[list[MLModel]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class MLModel(Base):
    __tablename__ = "ml_models"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_ml_model_name_version"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("training_experiments.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(100))
    version: Mapped[str] = mapped_column(String(40))
    task: Mapped[str] = mapped_column(String(40))
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    experiment: Mapped[TrainingExperiment | None] = relationship(back_populates="models")
    metrics: Mapped[list[ModelMetric]] = relationship(back_populates="model", cascade="all, delete-orphan")


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    __table_args__ = (UniqueConstraint("model_id", "partition", "metric_name", name="uq_model_metric_partition"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("ml_models.id", ondelete="CASCADE"), index=True)
    partition: Mapped[str] = mapped_column(String(24))
    metric_name: Mapped[str] = mapped_column(String(80))
    metric_value: Mapped[float] = mapped_column(Float)
    model: Mapped[MLModel] = relationship(back_populates="metrics")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("ml_models.id", ondelete="RESTRICT"), index=True)
    task: Mapped[str] = mapped_column(String(40))
    subject_reference: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    score: Mapped[float] = mapped_column(Float)
    predicted_class: Mapped[str] = mapped_column(String(64))
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    explanation_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Transaction(Base):
    __tablename__ = "transactions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True)
    external_reference: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    review_status: Mapped[str] = mapped_column(String(24), default="unreviewed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class CreditApplication(Base):
    __tablename__ = "credit_applications"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True)
    application_reference: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    review_status: Mapped[str] = mapped_column(String(24), default="unreviewed")
    # Applicant attributes are deliberately not persisted in this default schema.


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_actor_created", "actor_user_id", "created_at"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(100))
    entity_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
