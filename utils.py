
from pathlib import Path

import numpy as np
from PIL import Image


def load_dataset():
    with np.load("mnist.npz") as dataset:
        images = dataset["x_train"].astype("float32") / 255
        images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
        labels = np.eye(10)[dataset["y_train"]]
        return images, labels


def prepare_digit(image_path):
    image = Image.open(image_path).convert("L")
    pixels = np.array(image, dtype=np.uint8)
    border = np.concatenate((
        pixels[0, :], pixels[-1, :],
        pixels[:, 0], pixels[:, -1]
    ))

    if border.mean() > 127:
        pixels = 255 - pixels

    mask = pixels > 30
    if not mask.any():
        raise ValueError("No digit was found in the image")

    y, x = np.where(mask)
    cropped = Image.fromarray(pixels).crop(
        (x.min(), y.min(), x.max() + 1, y.max() + 1)
    )
    width, height = cropped.size
    scale = 20 / max(width, height)
    cropped = cropped.resize(
        (round(width * scale), round(height * scale)),
        Image.Resampling.LANCZOS
    )

    result = Image.new("L", (28, 28), 0)
    x_offset = (28 - cropped.width) // 2
    y_offset = (28 - cropped.height) // 2
    result.paste(cropped, (x_offset, y_offset))
    return np.array(result, dtype=np.float32).reshape(-1, 1) / 255


def image_paths(images_dir="images"):
    directory = Path(images_dir)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Image directory was not found: {directory.resolve()}"
        )

    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
