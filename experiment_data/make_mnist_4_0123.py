"""Build a four-class MNIST dataset for digits 0, 1, 2, and 3.

The raw IDX files in ``MNIST/raw`` are read without downloading anything. As
in the paper, each 28x28 image is center-cropped to 24x24 and average-pooled
to 4x4, then flattened to 16 features. Pixel values are scaled to [0, pi],
matching the value range of the existing ``mnist_4`` features.

All examples for the selected digits from both official MNIST splits are
combined and stratified into a configurable train/test split (80/20 by
default). This intentionally re-splits the combined source data to retain all
selected examples while honoring the requested ratio; the original official
train/test boundary is therefore not preserved. Labels use the same two-column
+1/-1 encoding pattern as the existing four-class dataset, in the requested
digit order. The dataset name must still be registered in the project's
loader/configuration before the training scripts can consume it by name.

Run from the repository root, for example:
    python experiment_data/make_mnist_4_0123.py
"""

import argparse
import gzip
import math
import struct
from pathlib import Path

import numpy as np


DEFAULT_DIGITS = (0, 1, 2, 3)


def _read_idx(raw_dir, file_stem, expected_magic):
    """Read an IDX file, preferring the uncompressed file when both exist."""
    raw_dir = Path(raw_dir)
    plain_path = raw_dir / file_stem
    compressed_path = raw_dir / f"{file_stem}.gz"

    if plain_path.is_file():
        data = plain_path.read_bytes()
    elif compressed_path.is_file():
        with gzip.open(compressed_path, "rb") as source:
            data = source.read()
    else:
        raise FileNotFoundError(
            f"Could not find {plain_path} or {compressed_path}. "
            "Expected the standard MNIST IDX files in the raw directory."
        )

    if len(data) < 4:
        raise ValueError(f"IDX file is shorter than its magic number: {file_stem}")

    magic = struct.unpack_from(">I", data, 0)[0]
    if magic != expected_magic:
        raise ValueError(
            f"Unexpected IDX magic number in {file_stem}: {magic}; "
            f"expected {expected_magic}."
        )

    dimension_count = magic & 0xFF
    header_end = 4 + dimension_count * 4
    if len(data) < header_end:
        raise ValueError(f"IDX header is incomplete: {file_stem}")

    shape = struct.unpack_from(f">{dimension_count}I", data, 4)
    expected_size = math.prod(shape)
    actual_size = len(data) - header_end
    if actual_size != expected_size:
        raise ValueError(
            f"IDX payload size mismatch in {file_stem}: "
            f"found {actual_size} bytes, expected {expected_size}."
        )

    return np.frombuffer(data, dtype=np.uint8, offset=header_end).reshape(shape).copy()


def load_raw_mnist(raw_dir):
    """Load standard MNIST image and label IDX files from ``raw_dir``."""
    train_images = _read_idx(raw_dir, "train-images-idx3-ubyte", 2051)
    train_labels = _read_idx(raw_dir, "train-labels-idx1-ubyte", 2049)
    test_images = _read_idx(raw_dir, "t10k-images-idx3-ubyte", 2051)
    test_labels = _read_idx(raw_dir, "t10k-labels-idx1-ubyte", 2049)

    if train_images.shape[0] != train_labels.shape[0]:
        raise ValueError("The MNIST training image and label counts do not match.")
    if test_images.shape[0] != test_labels.shape[0]:
        raise ValueError("The MNIST test image and label counts do not match.")

    return (
        np.concatenate((train_images, test_images), axis=0),
        np.concatenate((train_labels, test_labels), axis=0),
    )


def preprocess_images(images):
    """Center-crop 28x28 images to 24x24, then average-pool to 4x4."""
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Expected images with shape (N, 28, 28), got {images.shape}.")

    cropped = images[:, 2:26, 2:26].astype(np.float64, copy=False)
    pooled = cropped.reshape(-1, 4, 6, 4, 6).mean(axis=(2, 4))
    return (pooled.reshape(-1, 16) / 255.0) * np.pi


def encode_labels(labels, digits):
    """Encode four ordered digit classes as two bipolar (+1/-1) columns."""
    digit_to_class = {digit: class_index for class_index, digit in enumerate(digits)}
    encoded = np.empty((len(labels), 2), dtype=np.float64)

    for row_index, digit in enumerate(labels):
        class_index = digit_to_class[int(digit)]
        encoded[row_index, 0] = 1.0 if class_index < 2 else -1.0
        encoded[row_index, 1] = 1.0 if class_index % 2 == 0 else -1.0

    return encoded


def _parse_digits(value):
    try:
        digits = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Digits must be comma-separated integers.") from exc

    if len(digits) != 4 or len(set(digits)) != 4 or any(digit < 0 or digit > 9 for digit in digits):
        raise argparse.ArgumentTypeError("Specify exactly four distinct MNIST digits from 0 to 9.")
    return digits


def _split_by_class(images, labels, digits, train_ratio, seed):
    """Create a stratified split while retaining every selected example."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be greater than 0 and less than 1.")

    rng = np.random.default_rng(seed)
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    counts = {}

    for digit in digits:
        class_indices = np.flatnonzero(labels == digit)
        if len(class_indices) < 2:
            raise ValueError(f"At least two examples are required for digit {digit}.")

        class_indices = rng.permutation(class_indices)
        train_count = int(round(train_ratio * len(class_indices)))
        train_count = min(max(train_count, 1), len(class_indices) - 1)
        train_indices = class_indices[:train_count]
        test_indices = class_indices[train_count:]

        train_images.append(images[train_indices])
        train_labels.append(labels[train_indices])
        test_images.append(images[test_indices])
        test_labels.append(labels[test_indices])
        counts[digit] = (len(train_indices), len(test_indices))

    return (
        np.concatenate(train_images),
        np.concatenate(train_labels),
        np.concatenate(test_images),
        np.concatenate(test_labels),
        counts,
    )


def generate_dataset(raw_dir, out_dir, digits=DEFAULT_DIGITS, train_ratio=0.8, seed=42):
    """Generate and save the selected four-digit dataset as text files."""
    images, labels = load_raw_mnist(raw_dir)
    selected = np.isin(labels, digits)
    images = images[selected]
    labels = labels[selected]

    train_images, train_digits, test_images, test_digits, counts = _split_by_class(
        images, labels, digits, train_ratio, seed
    )

    x_train = preprocess_images(train_images)
    x_test = preprocess_images(test_images)
    y_train = encode_labels(train_digits, digits)
    y_test = encode_labels(test_digits, digits)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "x_train.txt", x_train, fmt="%.18e")
    np.savetxt(out_dir / "y_train.txt", y_train, fmt="%.18e")
    np.savetxt(out_dir / "x_test.txt", x_test, fmt="%.18e")
    np.savetxt(out_dir / "y_test.txt", y_test, fmt="%.18e")

    print(f"Saved {len(train_images) + len(test_images)} examples for digits {digits} to {out_dir}.")
    print(f"Train: {len(train_images)}; test: {len(test_images)}; train ratio: {train_ratio:.6g}")
    for digit in digits:
        train_count, test_count = counts[digit]
        print(f"Digit {digit}: train={train_count}, test={test_count}")
    print("Features: 16 per image, scaled to [0, pi]; labels: two bipolar columns.")


def main():
    default_raw_dir = Path(__file__).resolve().parent / "MNIST" / "raw"
    default_out_dir = Path(__file__).resolve().parent / "mnist_4_0123"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=default_raw_dir,
                        help="directory containing the standard MNIST IDX files")
    parser.add_argument("--out_dir", type=Path, default=default_out_dir,
                        help="directory where the four output .txt files are written")
    parser.add_argument("--digits", type=_parse_digits, default=DEFAULT_DIGITS,
                        help="four distinct comma-separated digit labels (default: 0,1,2,3)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="fraction assigned to training, split independently per class (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducible stratified splitting")
    args = parser.parse_args()

    generate_dataset(args.raw_dir, args.out_dir, args.digits, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()
