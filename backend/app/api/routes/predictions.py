from fastapi import APIRouter, HTTPException

from app.models.model_registry import list_models, score_model
from app.schemas.prediction import ModelScoreRequest

router = APIRouter(tags=["predictions"])


@router.get("/models", summary="List locally registered model versions")
def registered_models() -> dict[str, object]:
    return {"models": list_models()}


@router.post("/predictions/fraud", summary="Score one transaction with a registered fraud model")
def score_fraud(request: ModelScoreRequest) -> dict[str, object]:
    return _score(request, "fraud")


@router.post("/predictions/credit-risk", summary="Score one application with a registered credit model")
def score_credit_risk(request: ModelScoreRequest) -> dict[str, object]:
    return _score(request, "credit_risk")


def _score(request: ModelScoreRequest, task: str) -> dict[str, object]:
    try:
        return score_model(
            request.model_version,
            request.features,
            task=task,
            low_threshold=request.low_threshold,
            high_threshold=request.high_threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
