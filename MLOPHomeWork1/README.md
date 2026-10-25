# Fashion-MNIST MLOps Assignment

This project trains a Fashion-MNIST classifier with a LeNet-style CNN, tracks the run in MLflow, uploads the trained model artifact to MinIO, notifies a webhook after verified upload, and exposes inference through a FastAPI service and a Streamlit UI.

## Stack

- Python 3.11+
- PyTorch + torchvision
- MLflow
- MinIO
- FastAPI
- Streamlit
- Docker Compose

## Prerequisites

- Docker Desktop or Docker Engine
- Python 3.11 or newer


## Local development setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the project and dev dependencies:

   ```bash
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -e '.[dev]'
   ```

3. Start the required backing services with Docker Compose:

   ```bash
   docker compose up -d mlflow minio webhook
   ```

4. Train the model:

   ```bash
   python -m fashion_mnist.train --config configs/train.toml
   ```

5. Start the API locally:

   ```bash
   uvicorn fashion_mnist.api:app --host 0.0.0.0 --port 8000
   ```

6. Start the Streamlit UI locally:

   ```bash
   streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
   ```

## Docker Compose workflow

Build and start the full stack:

```bash
docker compose up --build
```

This starts the following services:

- MLflow UI
- MinIO object storage and console
- FastAPI inference API
- Webhook receiver
- Streamlit UI
- Training job container

The Compose file maps host ports as follows:

- MLflow UI: http://localhost:5001
- MinIO API: http://localhost:9003
- MinIO Console: http://localhost:9004
- Webhook: http://localhost:8003/webhook
- FastAPI: http://localhost:8004
- Streamlit UI: http://localhost:8503

Inside the Docker network, services use the internal ports:

- MLflow: http://mlflow:5000
- MinIO: minio:9000
- FastAPI: http://api:8000
- Webhook: http://webhook:8001/webhook

To run training in the Docker stack:

```bash
docker compose run --rm --no-deps train python -m fashion_mnist.train --config configs/train.toml
```

## Environment variables

The project reads runtime values from environment variables and `.env` files via Pydantic settings.

A typical local setup may include values such as:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
export MINIO_ENDPOINT=localhost:9003
export MINIO_ACCESS_KEY=XXX
export MINIO_SECRET_KEY=XXX
export MINIO_BUCKET=fashion-minst-bucket
export WEBHOOK_URL=http://localhost:8003/webhook
```

For local Docker-based execution, the Compose file already sets these values for the services.

## MLflow

- Tracking URI: http://localhost:5001
- Default experiment name: `fashion-mnist-lenet`
- Training runs log parameters, metrics, config files, and model artifact metadata.
- Model versions are associated with the MLflow run ID.

## MinIO

- Endpoint: localhost:9003
- Console: http://localhost:9004
- Default credentials: `XXXX` / `XXXX`
- Required bucket: `fashion-minst-bucket`
- Artifact pattern: `model/<mlflow-run-id>/model.pt`
- Latest manifest: `model/latest.json`

## Webhook

After the model is uploaded to MinIO and the MLflow run is finalized, the training pipeline sends a payload to the webhook endpoint:

- http://localhost:8003/webhook

The receiver prints the incoming JSON payload to stdout.

## Prediction API and UI

- Upload a single PNG, JPEG, or BMP image in the Streamlit UI.
- The app sends the file to the FastAPI `/predict` endpoint.
- The API preprocesses the image using the same grayscale-and-resize contract as training.
- The service loads the latest model from MinIO and returns the predicted class, confidence, probabilities, and model provenance.

## Tests

Run the automated tests with:

```bash
pytest -q
```

The tests are designed to mock external services rather than download large datasets or hit live cloud resources.

## Docker login credentials

Create the registry credentials file from the example:

```bash
cp docker/login.env.example docker/login.env
```

Then fill in your registry username and password, and run:

```bash
chmod +x docker/login.sh
./docker/login.sh
```

This script reads `DOCKER_REGISTRY`, `DOCKER_USERNAME`, and `DOCKER_PASSWORD` from `docker/login.env` and performs the login without committing secrets to source control.

## Notes

- Training and inference are intentionally separate code paths.
- The bucket spelling `fashion-minst-bucket` is preserved intentionally per the assignment.
- Credentials and downloaded artifacts are not committed to source control.
- Model versions are tracked by MLflow run ID.
