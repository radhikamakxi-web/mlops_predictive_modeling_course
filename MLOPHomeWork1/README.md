# Fashion-MNIST MLOps Assignment

This project trains a LeNet-5-style classifier on Fashion-MNIST, logs the run to MLflow, uploads the model artifact to MinIO in the required bucket `fashion-minst-bucket`, sends a completion webhook after verified upload, serves inference through FastAPI, and demonstrates image classification in a Streamlit UI.

## Stack

- Python 3.12
- PyTorch + torchvision
- MLflow
- MinIO
- FastAPI
- Streamlit
- Docker Compose

## Local setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python -m pip install -U pip setuptools wheel
   python -m pip install -e '.[dev]'
   ```
3. Copy environment variables:
   ```bash
   cp .env.example .env
   ```
4. Start local services:
   ```bash
   docker compose up -d mlflow minio webhook
   ```
5. Train the model:
   ```bash
   python -m fashion_mnist.train --config configs/train.toml
   ```
6. Start the inference API and UI:
   ```bash
   uvicorn fashion_mnist.api:app --host 0.0.0.0 --port 8000
   streamlit run ui/app.py
   ```

## Docker Compose workflow

```bash
docker compose up -d
```

This starts:
- MLflow UI at `http://localhost:5000`
- MinIO console at `http://localhost:9001`
- FastAPI at `http://localhost:8000`
- Streamlit at `http://localhost:8501`
- webhook receiver at `http://localhost:8001/webhook`

To train a model in the Docker stack:

```bash
docker compose run --rm train
```

The training pipeline writes a versioned model artifact to `fashion-minst-bucket/model/<run-id>/model.pt` and publishes the latest manifest at `model/latest.json`.

## MLflow

- Tracking URI: `http://localhost:5000`
- Experiment name defaults to `fashion-mnist-lenet` from `configs/train.toml`
- The training run logs parameters, metrics, config artifacts, and the model checkpoint artifact itself.

## MinIO

- Endpoint: `localhost:9000`
- Console: `http://localhost:9001`
- Login: `minioadmin` / `minioadmin`
- Required bucket: `fashion-minst-bucket`
- Model object key pattern: `model/<mlflow-run-id>/model.pt`
- Inference loads the latest published artifact from `model/latest.json`

## Webhook

After the model is uploaded to MinIO and the MLflow run is finalized, the training pipeline posts a payload to the webhook endpoint. The receiver is available at `http://localhost:8001/webhook` and prints the JSON payload to stdout.

## Prediction API and UI

- Upload a single PNG/JPEG/BMP image to the Streamlit UI.
- The app sends the file to the FastAPI `/predict` endpoint.
- The API preprocesses the image using the same grayscale-and-resize contract as training.
- The service loads the latest model from MinIO and returns the predicted class, confidence, probabilities, and model provenance.

## Tests

```bash
pytest -q
```

All provided tests are designed to mock external services rather than download datasets or touch real cloud resources.

## Notes

- Training and inference are intentionally separate.
- The bucket spelling `fashion-minst-bucket` is preserved intentionally per the assignment.
- Credentials are never committed to source control; use `.env` and environment variables in deployment.

## Docker login credentials

Create a local credentials file from the example and populate it with your registry username/password:

```bash
cp docker/login.env.example docker/login.env
# edit docker/login.env
```

Then log in without hardcoding secrets in source control:

```bash
chmod +x docker/login.sh
./docker/login.sh
```

The script reads `DOCKER_REGISTRY`, `DOCKER_USERNAME`, and `DOCKER_PASSWORD` from the config file and runs:

```bash
echo "$DOCKER_PASSWORD" | docker login "$DOCKER_REGISTRY" -u "$DOCKER_USERNAME" --password-stdin
```

This keeps registry credentials configurable per environment while avoiding accidental commits of secrets.
