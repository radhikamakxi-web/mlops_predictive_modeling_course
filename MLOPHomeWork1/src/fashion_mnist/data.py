"""One preprocessing contract for both dataset images and API uploads."""

import io
import warnings

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from fashion_mnist.config import TrainingConfig

PREPROCESSING = {"size": [28, 28], "mode": "L", "mean": [0.5], "std": [0.5]}
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
ALLOWED_FORMATS = {"PNG", "JPEG", "BMP"}
FASHION_MNIST_MIRROR = "https://fashion-mnist.s3.eu-central-1.amazonaws.com/"


def preprocess_image(content: bytes) -> torch.Tensor:
    if not content:
        raise ValueError("The uploaded image is empty")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise ValueError("Unsupported image format; use PNG, JPEG, or BMP")
                if image.width * image.height > 4_000_000:
                    raise ValueError("Image is too large; limit is 4 million pixels")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Upload a single, non-animated image")
                image.load()
                image = ImageOps.exif_transpose(image)
                if "A" in image.getbands() or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", image.size, (0, 0, 0, 255))
                    image = Image.alpha_composite(background, rgba).convert("RGB")
                return IMAGE_TRANSFORM(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid, corrupt, or excessively large image") from exc


def make_loaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    # Official S3 mirror avoids the original HTTP host's frequent 403 responses.
    datasets.FashionMNIST.mirrors = [FASHION_MNIST_MIRROR]
    dataset = datasets.FashionMNIST(
        root=config.data_dir, train=True, download=True, transform=IMAGE_TRANSFORM,
    )
    val_size = max(1, int(len(dataset) * config.validation_split))
    if val_size >= len(dataset):
        raise ValueError("Validation split leaves no training samples")
    train, val = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    if config.max_train_samples:
        train = Subset(train, range(min(config.max_train_samples, len(train))))
    if config.max_val_samples:
        val = Subset(val, range(min(config.max_val_samples, len(val))))
    common = {"batch_size": config.batch_size, "num_workers": config.num_workers}
    return (
        DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(config.seed),
                   **common),
        DataLoader(val, shuffle=False, **common),
    )
