from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.data.eda import profile_dataframe
from app.data.ingestion import DatasetValidationError, read_csv_bytes

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/profile", summary="Validate and profile an uploaded CSV")
async def profile_dataset(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
) -> dict[str, object]:
    contents = await file.read()
    try:
        frame = read_csv_bytes(contents, file.filename)
        return {"filename": file.filename, "profile": profile_dataframe(frame, target_column)}
    except (DatasetValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
