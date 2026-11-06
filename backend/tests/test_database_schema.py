from sqlalchemy import create_engine, inspect

from app.database.models import Base


def test_schema_creates_all_expected_tables_on_sqlite():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    names = set(inspect(engine).get_table_names())
    assert {"users", "datasets", "training_experiments", "ml_models", "model_metrics", "predictions", "transactions", "credit_applications", "audit_logs"} <= names
