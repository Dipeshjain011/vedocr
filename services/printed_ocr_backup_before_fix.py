import time
from functools import lru_cache

import numpy as np
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from services.image_preprocessor import preprocess_image

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


@lru_cache(maxsize=1)
def get_tesseract_version():
    """Check Tesseract and its English language model once per process."""
    version = str(pytesseract.get_tesseract_version())
    languages = pytesseract.get_languages(config="")
    if "eng" not in languages:
        raise RuntimeError("Tesseract English language model 'eng' is missing.")
    print(f"[OCR] Tesseract ready: {version}")
    return version


@lru_cache(maxsize=1)
def get_paddle_ocr():
    """Load PaddleOCR lazily for difficult scans."""
    from paddleocr import PaddleOCR

    print("[TIMING] Initializing PaddleOCR...")
    start = time.perf_counter()
    ocr = PaddleOCR(
        lang="en",
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv6_small_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    print(f"[TIMING] PaddleOCR initialized in {time.perf_counter() - start:.2f}s")
    return ocr


def _result(page, engine, status, text="", error=None):
    result = {
        "page": page.get("page", 1),
        "engine": engine,
        "status": status,
        "text": text,
    }
    if error:
        result["error"] = error
    return result


def _to_pil_image(image):
    """Convert PIL/NumPy input into an RGB image."""
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    try:
        array = np.asarray(image)
        if array.size == 0:
            raise ValueError("Image array is empty.")
        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    except Exception as exc:
        raise TypeError(f"Unsupported image type: {type(image).__name__}") from exc


def _ocr_variants(image):
    """Create limited OCR-friendly variants without destroying table lines."""
    image = _to_pil_image(image)
    width, height = image.size

    # Tesseract performs poorly on tiny text. Upscale modestly, but cap memory.
    if max(width, height) < 2200:
        scale = min(2.0, 2200 / max(width, height))
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    elif max(width, height) > 3000:
        scale = 3000 / max(width, height)
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    gray = ImageOps.grayscale(image)
    contrast = ImageEnhance.Contrast(ImageOps.autocontrast(gray)).enhance(1.25)
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    return [
        ("rgb", image),
        ("gray_autocontrast", contrast),
        ("sharpened", sharpened),
    ]


def _text_quality(text):
    """Heuristic ranking only; it does not validate clinical correctness."""
    if not text:
        return -1
    stripped = text.strip()
    if not stripped:
        return -1
    alnum = sum(ch.isalnum() for ch in stripped)
    letters = sum(ch.isalpha() for ch in stripped)
    score = min(len(stripped), 2500) * 0.01
    score += min(alnum, 1200) * 0.025
    score += min(letters, 900) * 0.015

    lower = stripped.casefold()
    # Reward useful report structure and table vocabulary.
    keywords = (
        "patient", "hospital", "doctor", "report", "lab results",
        "test name", "reference", "result", "hemoglobin", "glucose",
        "medications", "cetirizine", "vitals", "temperature",
    )
    score += sum(2.5 for word in keywords if word in lower)

    # Penalize long runs of repeated OCR debris.
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        debris = sum(
            1 for line in lines
            if len(line) <= 8 and not any(ch.isdigit() for ch in line)
            and not any(ch in line for ch in ":/%")
        )
        score -= min(debris, 20) * 0.7
    return score


def _run_tesseract(image):
    """Try a few page segmentation modes and image variants; return best text."""
    get_tesseract_version()
    candidates = []
    variants = _ocr_variants(image)

    # PSM 6 suits a uniform page; PSM 11 can help sparse tables/labels.
    for variant_name, variant in variants:
        for psm in (6, 11):
            try:
                text = pytesseract.image_to_string(
                    variant,
                    lang="eng",
                    config=f"--oem 1 --psm {psm}",
                    timeout=18,
                ).strip()
            except Exception as exc:
                print(f"[WARN] Tesseract {variant_name}/psm{psm}: {exc}")
                continue
            if text:
                candidates.append((_text_quality(text), text, variant_name, psm))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    print(
        f"[OCR] Best Tesseract variant={best[2]}, psm={best[3]}, "
        f"quality={best[0]:.1f}"
    )
    return best[1]


def _run_paddleocr(image):
    """Run PaddleOCR and normalize common 2.x/3.x result formats."""
    import json

    ocr = get_paddle_ocr()
    array = np.asarray(_to_pil_image(image))
    if callable(getattr(ocr, "predict", None)):
        prediction = ocr.predict(input=array)
    else:
        prediction = ocr.ocr(array, cls=False)

    def to_plain(value):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return to_plain(json.loads(value))
            except (json.JSONDecodeError, TypeError):
                return value
        if isinstance(value, dict):
            return {key: to_plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_plain(item) for item in value]
        json_value = getattr(value, "json", None)
        if json_value is not None:
            try:
                return to_plain(json_value() if callable(json_value) else json_value)
            except Exception:
                pass
        for method_name in ("to_dict", "dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    return to_plain(method())
                except Exception:
                    pass
        return str(value)

    def collect(value):
        output = []
        if isinstance(value, dict):
            for key in ("rec_texts", "texts", "text"):
                item = value.get(key)
                if isinstance(item, str) and item.strip():
                    output.append(item.strip())
                elif isinstance(item, (list, tuple)):
                    output.extend(
                        item_text.strip()
                        for item_text in item
                        if isinstance(item_text, str) and item_text.strip()
                    )
            if not output:
                for key in ("res", "result", "results", "data"):
                    if key in value:
                        output.extend(collect(value[key]))
        elif isinstance(value, (list, tuple)):
            for item in value:
                # PaddleOCR 2.x: [box, ("recognized text", confidence)]
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[1], (list, tuple))
                    and item[1]
                    and isinstance(item[1][0], str)
                ):
                    output.append(item[1][0].strip())
                else:
                    output.extend(collect(item))
        return output

    return "\n".join(collect(to_plain(prediction))).strip()


def _needs_second_engine(text):
    """Use PaddleOCR when Tesseract text is suspiciously sparse/corrupted."""
    if not text or len(text.strip()) < 80:
        return True
    lower = text.casefold()
    has_report_fields = any(
        term in lower for term in ("patient", "hospital", "doctor", "report")
    )
    has_lab_section = "lab results" in lower or "laboratory" in lower
    # If a lab heading is present but almost no useful text follows it,
    # a second OCR engine may recover table cells.
    if has_lab_section:
        lab_pos = max(lower.find("lab results"), lower.find("laboratory"))
        lab_chunk = lower[lab_pos:lab_pos + 700]
        useful = sum(ch.isalnum() for ch in lab_chunk)
        if useful < 45:
            return True
    return not has_report_fields


def extract_printed_text(page: dict) -> dict:
    """Use embedded text where appropriate; otherwise OCR with quality fallback."""
    total_start = time.perf_counter()
    page_number = page.get("page", 1)

    embedded_text = page.get("text", "")
    if not isinstance(embedded_text, str):
        embedded_text = ""
    embedded_text = embedded_text.strip()

    if page.get("kind") == "pdf_text" and embedded_text:
        return _result(page, "pymupdf_embedded_text", "text_extracted", embedded_text)

    image = page.get("image")
    if image is None:
        if embedded_text:
            return _result(page, "pymupdf_embedded_text", "text_extracted", embedded_text)
        return _result(
            page, "none", "setup_required",
            error="No image or embedded text was provided. Check document_loader.py.",
        )

    try:
        start = time.perf_counter()
        processed_image = preprocess_image(image)
        print(
            f"[TIMING] Page {page_number} preprocessing: "
            f"{time.perf_counter() - start:.2f}s"
        )
        if processed_image is None:
            raise ValueError("Image preprocessing returned no image.")

        start = time.perf_counter()
        try:
            recognized_text = _run_tesseract(processed_image)
        except Exception as exc:
            recognized_text = ""
            print(f"[WARN] Tesseract failed: {type(exc).__name__}: {exc}")
        print(
            f"[TIMING] Page {page_number} Tesseract: "
            f"{time.perf_counter() - start:.2f}s"
        )
        engine = "tesseract"

        # Do not run PaddleOCR for every page; use it when OCR appears weak.
        if _needs_second_engine(recognized_text):
            print(f"[OCR] Page {page_number}: Tesseract text looks weak; trying PaddleOCR")
            start = time.perf_counter()
            try:
                paddle_text = _run_paddleocr(processed_image)
                print(
                    f"[TIMING] Page {page_number} PaddleOCR: "
                    f"{time.perf_counter() - start:.2f}s"
                )
                if paddle_text and _text_quality(paddle_text) > _text_quality(recognized_text):
                    recognized_text = paddle_text
                    engine = "paddleocr"
            except Exception as exc:
                print(f"[WARN] PaddleOCR fallback failed: {type(exc).__name__}: {exc}")

        # Embedded text may supplement OCR only when the page is not a
        # scanned image; avoid duplicating an image-only OCR result.
        final_text = recognized_text or embedded_text
        if not final_text:
            return _result(
                page, engine, "no_text_detected",
                error="Neither OCR engine nor embedded text returned readable text.",
            )

        return _result(page, engine, "text_extracted", final_text)

    except Exception as exc:
        print(f"[ERROR] Page {page_number}: {type(exc).__name__}: {exc}")
        return _result(
            page, "tesseract", "error",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        print(
            f"[TIMING] Page {page_number} total OCR function time: "
            f"{time.perf_counter() - total_start:.2f}s"
        )
