
import json
import time
from functools import lru_cache
from typing import Any

from services.image_preprocessor import preprocess_image




@lru_cache(maxsize=1)
def get_paddle_ocr():
    """Load lightweight OCR models."""

    from paddleocr import PaddleOCR

    print("[TIMING] Initializing lightweight PaddleOCR...")
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

    print(
        f"[TIMING] Lightweight OCR initialized in "
        f"{time.perf_counter() - start:.2f}s"
    )

    return ocr



def _to_plain_data(value: Any) -> Any:
    """Convert PaddleOCR output to standard Python objects."""

    if value is None:
        return None

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return _to_plain_data(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        return value

    if isinstance(value, dict):
        return {
            key: _to_plain_data(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]

    # PaddleOCR 3.x result objects may expose JSON output.
    json_value = getattr(value, "json", None)

    if json_value is not None:
        try:
            if callable(json_value):
                json_value = json_value()

            return _to_plain_data(json_value)
        except Exception as exc:
            print(f"[DEBUG] Could not read result JSON: {exc}")

    for method_name in ("to_dict", "dict"):
        method = getattr(value, method_name, None)

        if callable(method):
            try:
                return _to_plain_data(method())
            except Exception:
                pass

    return str(value)


def _collect_text(data: Any) -> list[str]:
    """Extract recognized text from PaddleOCR 2.x or 3.x output."""

    collected = []

    if isinstance(data, dict):
        # PaddleOCR 3.x commonly stores recognized lines in rec_texts.
        for key in ("rec_texts", "texts", "text"):
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                collected.append(value.strip())

            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        collected.append(item.strip())

        if not collected:
            for key in ("res", "result", "results", "data"):
                if key in data:
                    collected.extend(_collect_text(data[key]))

        return collected

    if isinstance(data, (list, tuple)):
        for item in data:
            # PaddleOCR 2.x format:
            # [box, ("recognized text", confidence)]
            if (
                isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[1], (list, tuple))
                and len(item[1]) >= 1
                and isinstance(item[1][0], str)
            ):
                text = item[1][0].strip()

                if text:
                    collected.append(text)
            else:
                collected.extend(_collect_text(item))

    return collected


def _result(
    page: dict,
    engine: str,
    status: str,
    text: str = "",
    error: str | None = None,
) -> dict:
    """Create a consistent page result."""

    result = {
        "page": page.get("page", 1),
        "engine": engine,
        "status": status,
        "text": text,
    }

    if error:
        result["error"] = error

    return result


def extract_printed_text(page: dict) -> dict:
    """Extract text from embedded PDFs, scanned PDFs and images."""

    total_start = time.perf_counter()
    page_number = page.get("page", 1)

    embedded_text = page.get("text", "")

    if not isinstance(embedded_text, str):
        embedded_text = ""

    embedded_text = embedded_text.strip()

    # Use embedded text directly for text-based PDF pages.
    if page.get("kind") == "pdf_text" and embedded_text:
        return _result(
            page,
            "pymupdf_embedded_text",
            "text_extracted",
            embedded_text,
        )

    image = page.get("image")

    if image is None:
        if embedded_text:
            return _result(
                page,
                "pymupdf_embedded_text",
                "text_extracted",
                embedded_text,
            )

        return _result(
            page,
            "none",
            "setup_required",
            error=(
                "No image or embedded text was provided. "
                "Check document_loader.py."
            ),
        )

    try:
        # 1. Preprocess the image.
        start = time.perf_counter()
        processed_image = preprocess_image(image)

        print(
            f"[TIMING] Page {page_number} preprocessing: "
            f"{time.perf_counter() - start:.2f}s"
        )

        if processed_image is None:
            return _result(
                page,
                "paddleocr",
                "error",
                error="Image preprocessing returned no image.",
            )

        # 2. Get the cached OCR model.
        start = time.perf_counter()
        ocr = get_paddle_ocr()

        print(
            f"[TIMING] Page {page_number} model access: "
            f"{time.perf_counter() - start:.2f}s"
        )

        # 3. Run OCR.
        start = time.perf_counter()

        if callable(getattr(ocr, "predict", None)):
            prediction = ocr.predict(input=processed_image)

        elif callable(getattr(ocr, "ocr", None)):
            try:
                prediction = ocr.ocr(processed_image, cls=False)
            except TypeError:
                prediction = ocr.ocr(processed_image)

        else:
            return _result(
                page,
                "paddleocr",
                "error",
                error="No supported predict() or ocr() method found.",
            )

        print(
            f"[TIMING] Page {page_number} OCR inference: "
            f"{time.perf_counter() - start:.2f}s"
        )

        # 4. Inspect and convert OCR output.
        start = time.perf_counter()

        print("[DEBUG] Prediction type:", type(prediction))

        plain_data = _to_plain_data(prediction)

        print("[DEBUG] Converted output type:", type(plain_data))
        print("[DEBUG] Converted output:", repr(plain_data)[:3000])

        recognized_lines = _collect_text(plain_data)
        recognized_text = "\n".join(recognized_lines).strip()

        print("[DEBUG] Recognized lines:", recognized_lines)
        print("[DEBUG] Recognized text:", repr(recognized_text))

        print(
            f"[TIMING] Page {page_number} text conversion: "
            f"{time.perf_counter() - start:.2f}s"
        )

        # 5. Combine recognized text with any embedded PDF text.
        if embedded_text and recognized_text:
            final_text = embedded_text + "\n" + recognized_text
        elif recognized_text:
            final_text = recognized_text
        else:
            final_text = embedded_text

        if not final_text:
            return _result(
                page,
                "paddleocr",
                "no_text_detected",
                error=(
                    "PaddleOCR returned no extractable text. "
                    "Inspect the DEBUG output above."
                ),
            )

        return _result(
            page,
            "paddleocr",
            "text_extracted",
            final_text,
        )

    except ImportError as exc:
        return _result(
            page,
            "paddleocr_not_installed",
            "setup_required",
            error=str(exc),
        )

    except Exception as exc:
        print(
            f"[ERROR] Page {page_number}: "
            f"{type(exc).__name__}: {exc}"
        )

        return _result(
            page,
            "paddleocr",
            "error",
            error=f"{type(exc).__name__}: {exc}",
        )

    finally:
        print(
            f"[TIMING] Page {page_number} total OCR function time: "
            f"{time.perf_counter() - total_start:.2f}s"
        )
