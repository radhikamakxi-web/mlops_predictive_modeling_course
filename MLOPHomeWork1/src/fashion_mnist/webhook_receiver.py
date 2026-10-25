"""Local-only demo receiver: unauthenticated; do not expose to untrusted networks."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI(title="Fashion-MNIST completion webhook")


class CompletionPayload(BaseModel):
    """Training completion notification, allowing optional additional metadata."""

    model_config = ConfigDict(extra="allow")

    status: Literal["completed"]
    run_id: str = Field(min_length=1)
    model_bucket: str = Field(min_length=1)
    model_object_key: str = Field(min_length=1)
    validation_accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook")
def receive_webhook(payload: CompletionPayload) -> dict[str, bool | str]:
    # One complete JSON record per line, including any extra metadata.
    print(payload.model_dump_json(), flush=True)
    return {"received": True, "run_id": payload.run_id}