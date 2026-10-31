
import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image):
    """Prepare an image efficiently for PaddleOCR 3.x."""

    if image is None:
        raise ValueError("No image provided for preprocessing.")

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Input image array is empty.")

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Unsupported image array shape: {image.shape}"
            )

        image = Image.fromarray(image.astype(np.uint8))

    elif not isinstance(image, Image.Image):
        raise TypeError(
            f"Unsupported image type: {type(image).__name__}"
        )

    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size
    max_dimension = 1600

    # Downscale excessively large images to reduce inference time.
    # Never enlarge small images unnecessarily.
    largest_side = max(width, height)

    if largest_side > max_dimension:
        scale = max_dimension / largest_side

        new_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        )

        image = image.resize(
            new_size,
            Image.Resampling.BILINEAR,
        )

    return np.asarray(image, dtype=np.uint8)
