from pydantic import BaseModel, Field


class ModelScoreRequest(BaseModel):
    model_version: str
    features: dict[str, object]
    low_threshold: float = Field(default=0.2, ge=0, le=1)
    high_threshold: float = Field(default=0.75, ge=0, le=1)
