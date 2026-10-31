
from io import BytesIO

import fitz
from PIL import Image, UnidentifiedImageError

from config import MAX_PDF_PAGES


def load_upload_bytes(content: bytes, suffix: str) -> list[dict]:
    """Load PDF pages or uploaded images for OCR processing."""

    if suffix.lower() == ".pdf":
        output = []

        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                if len(pdf) > MAX_PDF_PAGES:
                    raise ValueError(
                        f"PDF exceeds {MAX_PDF_PAGES} pages."
                    )

                for page in pdf:
                    embedded_text = page.get_text("text").strip()
                    has_images = bool(page.get_images(full=True))

                    # Scanned/image-based page: preserve the image
                    # even when the page contains a small text footer.
                    if has_images:
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(3, 3),
                            alpha=False,
                        )

                        image = Image.open(
                            BytesIO(pix.tobytes("png"))
                        ).convert("RGB")

                        output.append({
                            "kind": "image",
                            "image": image,
                            "text": embedded_text,
                            "page": page.number + 1,
                        })

                    elif embedded_text:
                        # Ordinary text-based PDF page.
                        output.append({
                            "kind": "pdf_text",
                            "text": embedded_text,
                            "page": page.number + 1,
                        })

                    else:
                        # Page has neither usable text nor embedded
                        # images; render it anyway for OCR.
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(3, 3),
                            alpha=False,
                        )

                        image = Image.open(
                            BytesIO(pix.tobytes("png"))
                        ).convert("RGB")

                        output.append({
                            "kind": "image",
                            "image": image,
                            "text": "",
                            "page": page.number + 1,
                        })

        except fitz.FileDataError as exc:
            raise ValueError(
                "Cannot read PDF; it may be damaged or password-protected."
            ) from exc

        return output

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
        return [{
            "kind": "image",
            "image": image,
            "text": "",
            "page": 1,
        }]

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Cannot read uploaded image.") from exc
