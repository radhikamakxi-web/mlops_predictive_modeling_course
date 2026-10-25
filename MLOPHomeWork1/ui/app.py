"""Local Streamlit client; use a trusted API endpoint before exposing publicly."""

from __future__ import annotations

import math
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


API_URL = os.getenv("API_URL", "http://localhost:8000").strip().rstrip("/")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
STATUS_TIMEOUT = (3, 10)
PREDICTION_TIMEOUT = (5, 60)
IMAGE_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "BMP": "image/bmp"}


def api_detail(response: requests.Response) -> str:
    """Extract useful API errors without displaying HTML error pages."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: the API did not return a JSON error."
    detail = body.get("detail", body.get("message")) if isinstance(body, dict) else None
    return f"HTTP {response.status_code}: {str(detail)[:600]}" if detail else f"HTTP {response.status_code}"


def check_readiness() -> bool:
    """A live process is not necessarily ready to serve a trained model."""
    try:
        health = requests.get(f"{API_URL}/health", timeout=STATUS_TIMEOUT)
        if health.status_code != 200:
            st.error(f"The prediction service is not healthy. {api_detail(health)}")
            return False
        ready = requests.get(f"{API_URL}/ready", timeout=STATUS_TIMEOUT)
    except requests.Timeout:
        st.warning("The API status check timed out. Check the service and try Refresh status.")
        return False
    except requests.RequestException:
        st.error("Cannot reach the prediction API. Start the API and check API_URL and network connectivity.")
        return False

    if ready.status_code == 200:
        st.success("Model ready · Upload an image to get started.")
        return True
    if ready.status_code == 503:
        st.info(
            "The API is running, but no model is available yet. If training has not run, "
            "complete a successful training run and model upload first. Otherwise, check "
            "the API's model storage configuration, then refresh. Training is not run by this page."
        )
        st.caption(api_detail(ready))
    else:
        st.error(f"Could not check model readiness. {api_detail(ready)}")
    return False


def validate_prediction(data: object) -> dict:
    """Reject malformed API output rather than showing misleading results."""
    def probability(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        )

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    if type(data.get("class_id")) is not int or not 0 <= data["class_id"] <= 9:
        raise ValueError("Missing or invalid class_id.")
    if not isinstance(data.get("class_name"), str) or not data["class_name"].strip():
        raise ValueError("Missing or invalid class_name.")
    if not probability(data.get("confidence")):
        raise ValueError("Missing or invalid confidence.")
    probabilities = data.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities or not all(
        isinstance(name, str) and bool(name.strip()) and probability(value)
        for name, value in probabilities.items()
    ):
        raise ValueError("Missing or invalid class probabilities.")
    model = data.get("model")
    if not isinstance(model, dict) or not all(
        isinstance(model.get(key), str) and bool(model[key].strip())
        for key in ("bucket", "object_key", "run_id")
    ):
        raise ValueError("Missing or invalid model provenance.")
    return data


def predict(filename: str, content: bytes, mime_type: str) -> dict | None:
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (filename, content, mime_type)},
            timeout=PREDICTION_TIMEOUT,
        )
    except requests.Timeout:
        st.error("Prediction timed out. The service may still be loading its model; try again shortly.")
        return None
    except requests.RequestException:
        st.error("Prediction API connection failed. Check that the API is running and API_URL is correct.")
        return None
    if response.status_code != 200:
        if response.status_code == 503:
            st.warning("The model is currently unavailable. Complete training and model upload, or refresh status.")
        st.error(f"Prediction failed. {api_detail(response)}")
        return None
    try:
        return validate_prediction(response.json())
    except ValueError as exc:
        st.error(f"The API returned an invalid prediction response: {exc}")
        return None


def main() -> None:
    st.set_page_config(page_title="Fashion-MNIST Studio", page_icon="👕", layout="wide")
    st.markdown(
        """
        <style>
          .block-container { max-width: 1120px; padding-top: 2.5rem; }
          .studio-eyebrow { color: #7c83fd; font-weight: 700;
                            letter-spacing: .15em; font-size: .78rem; }
          h1 { letter-spacing: -.045em; }
          div[data-testid="stFileUploader"] {
            border: 1px dashed #818cf8; border-radius: 16px; padding: .5rem;
          }
          div[data-testid="stMetric"] {
            background: rgba(129, 140, 248, .09); border-radius: 14px; padding: 1rem;
          }
          .stButton > button { border-radius: 10px; }
        </style>
        <div class="studio-eyebrow">FASHION-MNIST / IMAGE CLASSIFICATION</div>
        """,
        unsafe_allow_html=True,
    )
    st.title("A little image. A clear prediction.")
    st.write("Upload one clothing image and explore the model's confidence across Fashion-MNIST classes.")
    with st.sidebar:
        st.subheader("Connection")
        st.caption("Prediction API")
        st.code(API_URL, language=None)
        st.button("Refresh status", use_container_width=True)
        st.divider()
        st.subheader("For best results")
        st.write("Use white or light clothing on a black or dark background, with one centered item.")
        st.write("The API automatically converts the image to grayscale and resizes it to 28 × 28 pixels.")
        st.caption("Fashion-MNIST-style images work best. Everyday photos may produce unreliable predictions.")
        st.caption("Images are sent to the configured API only when you click Classify image. No model runs in this UI.")

    ready = check_readiness()
    st.divider()
    upload_column, result_column = st.columns([1, 1.2], gap="large")
    with upload_column:
        st.subheader("01 / Your image")
        uploaded = st.file_uploader(
            "Choose a clothing image",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=False,
            help="One PNG, JPEG, or BMP image, up to 5 MB.",
        )
        st.caption("PNG, JPEG or BMP · One image · Maximum 5 MB")
        content = None
        mime_type = None
        if uploaded is not None:
            if uploaded.size > MAX_UPLOAD_BYTES:
                st.error("This image exceeds the 5 MB limit. Choose a smaller file.")
            elif uploaded.size == 0:
                st.error("This file is empty. Choose a valid image.")
            else:
                try:
                    image_bytes = uploaded.getvalue()
                    with Image.open(BytesIO(image_bytes)) as image:
                        mime_type = IMAGE_TYPES.get(image.format)
                        if mime_type is None:
                            raise ValueError("Only PNG, JPEG, and BMP images are supported.")
                        if image.width * image.height > 25_000_000:
                            raise ValueError("Image dimensions are too large. Use an image under 25 megapixels.")
                        image.verify()
                    st.image(image_bytes, caption=uploaded.name, width=280)
                    content = image_bytes
                except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
                    st.error("Cannot preview this image. Choose an intact PNG, JPEG, or BMP under 25 megapixels.")
        classify = st.button(
            "Classify image", type="primary", use_container_width=True,
            disabled=not ready or content is None,
        )

    with result_column:
        st.subheader("02 / Prediction")
        if not classify:
            st.info("Your predicted class, confidence, and probability chart will appear here.")
            return
        with st.spinner("Asking the model…"):
            result = predict(uploaded.name, content, mime_type)
        if result is None:
            return
        label, confidence = st.columns(2)
        label.metric("Predicted class", result["class_name"])
        confidence.metric("Confidence", f"{result['confidence']:.1%}")
        st.caption(f"Class ID: {result['class_id']} · Confidence is a model estimate, not a guarantee.")
        st.markdown("#### Class probabilities")
        ordered = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
        st.bar_chart(
            {"Class": [name for name, _ in ordered], "Probability": [value for _, value in ordered]},
            x="Class", y="Probability", color="#818cf8",
        )
        with st.expander("Model provenance"):
            st.json(result["model"])


if __name__ == "__main__":
    main()