from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.data.ingestion import DatasetValidationError, read_csv_bytes
from app.models.clustering.analyzer import ALGORITHMS, discover_patterns

router = APIRouter(prefix="/clustering", tags=["clustering"])


@router.post("/discover", summary="Find unsupervised transaction patterns")
async def clustering_discover(
    file: UploadFile = File(...),
    algorithm: str = Form(default="kmeans"),
    reducer: str = Form(default="pca"),
    n_clusters: int = Form(default=4),
    eps: float = Form(default=0.5),
    min_samples: int = Form(default=5),
    target_column: str | None = Form(default=None),
) -> dict[str, object]:
    contents = await file.read()
    try:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unsupported algorithm. Choose: {', '.join(ALGORITHMS)}")
        frame = read_csv_bytes(contents, file.filename)
        excluded = (target_column,) if target_column else ()
        return discover_patterns(frame, algorithm=algorithm, reducer=reducer, n_clusters=n_clusters, eps=eps, min_samples=min_samples, exclude_columns=excluded)
    except (DatasetValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
