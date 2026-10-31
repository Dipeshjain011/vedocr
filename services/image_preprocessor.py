import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


def preprocess_image(image):
    """
    Prepare a scanned page for OCR while preserving table structure.

    Returns an RGB uint8 NumPy array. Keep this preprocessing conservative:
    aggressive thresholding can erase faint grid lines and small characters.
    Multiple OCR-friendly variants are generated in printed_ocr.py.
    """
    if image is None:
        raise ValueError("No image provided for preprocessing.")

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Input image array is empty.")
        if image.ndim == 2:
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 1:
            image = Image.fromarray(image[:, :, 0].astype(np.uint8)).convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 4:
            image = Image.fromarray(image[:, :, :3].astype(np.uint8)).convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 3:
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image array shape: {image.shape}")
    elif isinstance(image, Image.Image):
        image = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    image = ImageOps.exif_transpose(image).convert("RGB")

    # Correct low-contrast scans without converting to harsh black/white.
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.15)
    gray = gray.filter(ImageFilter.SHARPEN)

    # Tesseract and PaddleOCR both accept RGB; return RGB for compatibility.
    image = gray.convert("RGB")

    width, height = image.size
    max_dimension = 2600
    largest_side = max(width, height)
    if largest_side > max_dimension:
        scale = max_dimension / largest_side
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    return np.asarray(image, dtype=np.uint8)
