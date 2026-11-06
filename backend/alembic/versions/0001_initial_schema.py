"""Create the initial FinRisk metadata schema."""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table(
        "datasets",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("target_column", sa.String(128)),
        sa.Column("time_column", sa.String(128)),
        sa.Column("row_count", sa.Integer()),
        sa.Column("schema_summary", sa.JSON()),
        sa.Column("content_sha256", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_datasets_content_sha256", "datasets", ["content_sha256"])
    op.create_table(
        "training_experiments",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("dataset_id", sa.Uuid(), sa.ForeignKey("datasets.id", ondelete="SET NULL")),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("task", sa.String(40), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("split_strategy", sa.String(40)),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "ml_models",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("experiment_id", sa.Uuid(), sa.ForeignKey("training_experiments.id", ondelete="SET NULL")),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("version", sa.String(40), nullable=False),
        sa.Column("task", sa.String(40), nullable=False),
        sa.Column("artifact_uri", sa.Text()),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", "version", name="uq_ml_model_name_version"),
    )
    op.create_table(
        "model_metrics",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_id", sa.Uuid(), sa.ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("partition", sa.String(24), nullable=False),
        sa.Column("metric_name", sa.String(80), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.UniqueConstraint("model_id", "partition", "metric_name", name="uq_model_metric_partition"),
    )
    op.create_index("ix_model_metrics_model_id", "model_metrics", ["model_id"])
    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_id", sa.Uuid(), sa.ForeignKey("ml_models.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("task", sa.String(40), nullable=False),
        sa.Column("subject_reference", sa.String(128)),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("predicted_class", sa.String(64), nullable=False),
        sa.Column("threshold", sa.Float()),
        sa.Column("explanation_summary", sa.JSON()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_predictions_model_id", "predictions", ["model_id"])
    op.create_index("ix_predictions_subject_reference", "predictions", ["subject_reference"])
    op.create_table(
        "transactions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("prediction_id", sa.Uuid(), sa.ForeignKey("predictions.id", ondelete="SET NULL")),
        sa.Column("external_reference", sa.String(128)),
        sa.Column("occurred_at", sa.DateTime(timezone=True)),
        sa.Column("amount", sa.Float()),
        sa.Column("currency", sa.String(3)),
        sa.Column("review_status", sa.String(24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_transactions_prediction_id", "transactions", ["prediction_id"])
    op.create_index("ix_transactions_external_reference", "transactions", ["external_reference"])
    op.create_table(
        "credit_applications",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("prediction_id", sa.Uuid(), sa.ForeignKey("predictions.id", ondelete="SET NULL")),
        sa.Column("application_reference", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("review_status", sa.String(24), nullable=False),
    )
    op.create_index("ix_credit_applications_prediction_id", "credit_applications", ["prediction_id"])
    op.create_index("ix_credit_applications_application_reference", "credit_applications", ["application_reference"])
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("actor_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(80)),
        sa.Column("entity_id", sa.String(128)),
        sa.Column("details", sa.JSON()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_logs_actor_user_id", "audit_logs", ["actor_user_id"])
    op.create_index("ix_audit_actor_created", "audit_logs", ["actor_user_id", "created_at"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("credit_applications")
    op.drop_table("transactions")
    op.drop_table("predictions")
    op.drop_table("model_metrics")
    op.drop_table("ml_models")
    op.drop_table("training_experiments")
    op.drop_table("datasets")
    op.drop_table("users")
