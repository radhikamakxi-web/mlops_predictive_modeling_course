"""Single-image inference API. Interactive documentation is served at /docs."""

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from fashion_mnist.config import Settings
from fashion_mnist.data import preprocess_image
from fashion_mnist.inference import ModelUnavailable, Predictor


class ModelReference(BaseModel):
    bucket: str
    object_key: str
    run_id: str


class PredictionResponse(BaseModel):
    class_id: int = Field(ge=0, le=9)
    class_name: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model: ModelReference


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    settings = settings or Settings()
    service = predictor if predictor is not None else Predictor(settings)
    application = FastAPI(title="Fashion-MNIST Prediction API", version="1.0.0")
    application.state.predictor = service

    @application.get("/health")
    def health():
        """Liveness: API process is running, even when no model exists yet."""
        return {"status": "ok"}

    @application.get("/ready")
    def ready():
        """Readiness: the selected model can be loaded from MinIO."""
        try:
            return {"status": "ready", "model": service.ensure_loaded()}
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        """Upload one PNG, JPEG, or BMP as multipart/form-data using field 'file'."""
        try:
            content = await file.read(settings.max_upload_bytes + 1)
        finally:
            await file.close()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Image exceeds the upload size limit (5 MiB by default)")
        try:
            image = await run_in_threadpool(preprocess_image, content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            return await run_in_threadpool(service.predict, image)
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return application


app = create_app()
