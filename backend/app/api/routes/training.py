from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError

from app.data.ingestion import DatasetValidationError, read_csv_bytes
from app.models.machine_learning.classifiers import MODEL_NAMES
from app.services.training import train_and_evaluate
from app.core.config import get_settings
from app.services.persistence import persist_training_run

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/evaluate", summary="Train a baseline classifier and score its validation partition")
async def evaluate_training_run(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    model_name: str = Form(default="random_forest"),
    balance_strategy: str = Form(default="none"),
    time_column: str | None = Form(default=None),
    include_explanations: bool = Form(default=False),
    feature_selection: str = Form(default="none"),
    task: str = Form(default="fraud"),
    save_model: bool = Form(default=True),
    positive_label: str | None = Form(default=None),
) -> dict[str, object]:
    contents = await file.read()
    try:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unsupported model. Choose one of: {', '.join(MODEL_NAMES)}")
        frame = read_csv_bytes(contents, file.filename)
        result = train_and_evaluate(
            frame,
            target_column=target_column,
            model_name=model_name,
            balance_strategy=balance_strategy,
            time_column=time_column,
            include_explanations=include_explanations,
            feature_selection=feature_selection,
            task=task,
            save_model=save_model,
            positive_label=positive_label,
        )
        result["persistence"] = {"saved": False, "raw_rows_saved": False}
        if get_settings().persistence_enabled:
            try:
                result["persistence"] = persist_training_run(
                    filename=file.filename or "upload.csv",
                    csv_bytes=contents,
                    frame=frame,
                    target_column=target_column,
                    model_name=model_name,
                    balance_strategy=balance_strategy,
                    feature_selection=feature_selection,
                    result=result,
                )
            except SQLAlchemyError:
                result["persistence"] = {"saved": False, "raw_rows_saved": False, "reason": "Database unavailable; result returned without persistence."}
        return result
    except (DatasetValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()
