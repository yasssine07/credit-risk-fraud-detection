from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/health", summary="Check API process health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/platform", summary="Describe configured platform capabilities")
def platform_info() -> dict[str, object]:
    return {
        "name": "FinRisk AI",
        "stage": "phase-1-foundation",
        "capabilities": ["credit-risk", "fraud-detection", "pattern-discovery"],
        "model_ready": False,
        "dataset_configured": False,
    }
