FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY ui ./ui
COPY configs ./configs

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e . \
    && pip install boto3

EXPOSE 8000 8001 8501

CMD ["python", "-m", "uvicorn", "fashion_mnist.api:app", "--host", "0.0.0.0", "--port", "8000"]
