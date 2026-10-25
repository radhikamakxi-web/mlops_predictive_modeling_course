"""Export one untouched held-out test image per class for UI/API demonstrations."""

import argparse
from pathlib import Path

from torchvision.datasets import FashionMNIST

from fashion_mnist.data import FASHION_MNIST_MIRROR
from fashion_mnist.model import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="samples")
    args = parser.parse_args()
    FashionMNIST.mirrors = [FASHION_MNIST_MIRROR]
    dataset = FashionMNIST(args.data_dir, train=False, download=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    for image, label in dataset:
        if label not in seen:
            name = CLASS_NAMES[label].lower().replace("/", "-").replace(" ", "-")
            image.save(output / f"{label}-{name}.png")
            seen.add(label)
        if len(seen) == len(CLASS_NAMES):
            break
    print(f"Saved {len(seen)} held-out test images to {output}")


if __name__ == "__main__":
    main()