"""Image preprocessing and dataset contract without downloading Fashion-MNIST."""

import io

import pytest
import torch
from PIL import Image, ImageOps

from fashion_mnist import data
from fashion_mnist.config import TrainingConfig
from fashion_mnist.model import CLASS_NAMES, LeNet5


def encode(image, image_format="PNG", **kwargs):
    output = io.BytesIO()
    image.save(output, format=image_format, **kwargs)
    return output.getvalue()


@pytest.mark.parametrize("batch_size", [1, 3])
def test_model_output_shape_and_backward(batch_size):
    model = LeNet5()
    logits = model(torch.zeros(batch_size, 1, 28, 28))
    assert logits.shape == (batch_size, 10)
    assert len(CLASS_NAMES) == 10
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("pixel", [0, 128, 255])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "BMP"])
def test_grayscale_shape_dtype_and_normalization(pixel, image_format):
    result = data.preprocess_image(encode(Image.new("L", (35, 19), pixel), image_format))
    assert result.shape == (1, 1, 28, 28)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.full_like(result, pixel / 127.5 - 1))


def test_rgb_is_converted_to_grayscale():
    image = Image.new("RGB", (28, 28), (255, 0, 0))
    result = data.preprocess_image(encode(image))
    expected = image.convert("L").getpixel((0, 0)) / 127.5 - 1
    torch.testing.assert_close(result, torch.full_like(result, expected))


def test_transparency_composites_onto_black():
    result = data.preprocess_image(encode(Image.new("RGBA", (28, 28), (255, 255, 255, 0))))
    torch.testing.assert_close(result, -torch.ones_like(result))


def test_exif_orientation_is_applied():
    image = Image.new("RGB", (40, 20), "black")
    image.paste("white", (0, 0, 20, 20))
    exif = Image.Exif()
    exif[274] = 6
    content = encode(image, "JPEG", exif=exif)
    with Image.open(io.BytesIO(content)) as decoded:
        expected = data.IMAGE_TRANSFORM(ImageOps.exif_transpose(decoded)).unsqueeze(0)
    torch.testing.assert_close(data.preprocess_image(content), expected)


@pytest.mark.parametrize("content,match", [
    (b"", "empty"), (b"not an image", "Invalid"),
    (b"\x89PNG\r\n\x1a\n", "Invalid"),
])
def test_empty_invalid_and_corrupt_images(content, match):
    with pytest.raises(ValueError, match=match):
        data.preprocess_image(content)


@pytest.mark.parametrize("image_format", ["GIF", "TIFF"])
def test_unsupported_image_formats(image_format):
    with pytest.raises(ValueError, match="Unsupported"):
        data.preprocess_image(encode(Image.new("L", (28, 28)), image_format))


def test_oversize_image_is_rejected():
    content = encode(Image.new("L", (2001, 2000)))
    with pytest.raises(ValueError, match="4 million pixels"):
        data.preprocess_image(content)


@pytest.mark.parametrize("limit", [300, 100])
def test_decompression_bomb_warning_and_error_become_value_errors(monkeypatch, limit):
    content = encode(Image.new("L", (20, 20)))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", limit)
    with pytest.raises(ValueError, match="excessively large"):
        data.preprocess_image(content)


def test_animated_png_is_rejected():
    first = Image.new("RGB", (28, 28), "black")
    second = Image.new("RGB", (28, 28), "white")
    content = encode(first, save_all=True, append_images=[second], duration=100, loop=0)
    with pytest.raises(ValueError, match="single, non-animated"):
        data.preprocess_image(content)


def test_loaders_share_inference_preprocessing_and_reproducible_split(monkeypatch, tmp_path):
    calls = []

    class FakeFashionMNIST:
        mirrors = []

        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.transform = kwargs["transform"]

        def __len__(self):
            return 20

        def __getitem__(self, index):
            image = Image.new("L", (28, 28), index * 10)
            return self.transform(image), index

    monkeypatch.setattr(data.datasets, "FashionMNIST", FakeFashionMNIST)
    config = TrainingConfig(
        data_dir=str(tmp_path), batch_size=3, validation_split=0.25,
        max_train_samples=7, max_val_samples=3,
    )
    train, val = data.make_loaders(config)
    train_again, val_again = data.make_loaders(config)
    assert len(train.dataset) == 7
    assert len(val.dataset) == 3
    assert calls[0] == {
        "root": str(tmp_path), "train": True, "download": True,
        "transform": data.IMAGE_TRANSFORM,
    }
    assert FakeFashionMNIST.mirrors == [data.FASHION_MNIST_MIRROR]
    train_ids = {label for _, label in train.dataset}
    val_ids = {label for _, label in val.dataset}
    assert train_ids.isdisjoint(val_ids)
    for left, right in [(train, train_again), (val, val_again)]:
        assert [labels.tolist() for _, labels in left] == [labels.tolist() for _, labels in right]
    for image, label in list(train.dataset) + list(val.dataset):
        inference = data.preprocess_image(encode(Image.new("L", (28, 28), label * 10)))
        torch.testing.assert_close(image, inference.squeeze(0), rtol=0, atol=0)


def test_split_requires_training_samples(monkeypatch):
    class OneImageDataset:
        mirrors = []

        def __init__(self, **kwargs):
            pass

        def __len__(self):
            return 1

    monkeypatch.setattr(data.datasets, "FashionMNIST", OneImageDataset)
    with pytest.raises(ValueError, match="no training samples"):
        data.make_loaders(TrainingConfig())