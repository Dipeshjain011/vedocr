import time
import re
from functools import lru_cache
import numpy as np
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from services.image_preprocessor import preprocess_image
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
@lru_cache(maxsize=1)
def get_tesseract_version():
    version = str(pytesseract.get_tesseract_version())
    languages = pytesseract.get_languages(config="")
    if "eng" not in languages:
        raise RuntimeError("Tesseract English language model is missing.")
    print(f"[OCR] Tesseract ready: {version}")
    return version
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
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("Image is empty.")
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")
def _ocr_variants(image):
    image = _to_pil_image(image)
    width, height = image.size
    if max(width, height) < 2200:
        scale = min(2.0, 2200 / max(width, height))
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.LANCZOS,
        )
    elif max(width, height) > 3000:
        scale = 3000 / max(width, height)
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.LANCZOS,
        )
    gray = ImageOps.grayscale(image)
    contrast = ImageEnhance.Contrast(
        ImageOps.autocontrast(gray)
    ).enhance(1.25)
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    return [
        ("rgb", image),
        ("gray", contrast),
        ("sharpened", sharpened),
    ]
def _text_quality(text):
    if not text or not text.strip():
        return -1
    text = text.strip()
    lower = text.casefold()
    score = min(len(text), 2500) * 0.01
    score += sum(ch.isalnum() for ch in text) * 0.025
    score += sum(ch.isalpha() for ch in text) * 0.015
    keywords = [
        "patient", "hospital", "doctor", "report",
        "lab results", "test name", "reference",
        "result", "hemoglobin", "glucose",
        "medications", "cetirizine", "vitals",
    ]
    score += sum(2.5 for word in keywords if word in lower)
    return score
def _run_tesseract(image):
    get_tesseract_version()
    candidates = []
    for variant_name, variant in _ocr_variants(image):
        for psm in (6, 11):
            try:
                text = pytesseract.image_to_string(
                    variant,
                    lang="eng",
                    config=f"--oem 1 --psm {psm}",
                    timeout=20,
                ).strip()
                if text:
                    candidates.append(
                        (_text_quality(text), text, variant_name, psm)
                    )
            except Exception as exc:
                print(f"[WARN] Tesseract {variant_name}/PSM {psm}: {exc}")
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    print(
        f"[OCR] Best variant={best[2]}, PSM={best[3]}, "
        f"quality={best[0]:.1f}"
    )
    return best[1]
def _run_paddleocr(image):
    """Optional fallback for difficult scanned pages."""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        array = np.asarray(_to_pil_image(image))
        if callable(getattr(ocr, "predict", None)):
            prediction = ocr.predict(input=array)
        else:
            prediction = ocr.ocr(array, cls=False)
        texts = []
        def collect(value):
            if isinstance(value, dict):
                for key in ("rec_texts", "texts"):
                    items = value.get(key)
                    if isinstance(items, list):
                        texts.extend(
                            str(item).strip()
                            for item in items
                            if str(item).strip()
                        )
                    elif isinstance(items, str) and items.strip():
                        texts.append(items.strip())
                if not texts:
                    for key in ("res", "result", "results", "data"):
                        if key in value:
                            collect(value[key])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if (
                        isinstance(item, (list, tuple))
                        and len(item) >= 2
                        and isinstance(item[1], (list, tuple))
                        and len(item[1]) > 0
                        and isinstance(item[1][0], str)
                    ):
                        texts.append(item[1][0].strip())
                    else:
                        collect(item)
            elif hasattr(value, "json"):
                raw = value.json
                if callable(raw):
                    raw = raw()
                collect(raw)
        collect(prediction)
        return "\n".join(texts).strip()
    except Exception as exc:
        print(f"[WARN] PaddleOCR failed: {exc}")
        return ""
def _extract_lab_table_text(image):
    """
    Detect the printed lab table's grid and OCR each cell independently.
    Values are read from the supplied image; none are hardcoded.
    """
    try:
        import cv2
        page_image = _to_pil_image(image)
        if max(page_image.size) < 2200:
            scale = min(3.0, 2200 / max(page_image.size))
            page_image = page_image.resize(
                (
                    int(page_image.width * scale),
                    int(page_image.height * scale),
                ),
                Image.Resampling.LANCZOS,
            )
        data = pytesseract.image_to_data(
            page_image,
            lang="eng",
            config="--oem 1 --psm 11",
            output_type=pytesseract.Output.DICT,
            timeout=20,
        )
        words = []
        for i, raw in enumerate(data["text"]):
            word = (raw or "").strip()
            if not word:
                continue
            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                confidence = 0
            if confidence < 5:
                continue
            words.append({
                "text": word,
                "lower": word.casefold().strip(".,:;"),
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
            })
# Find the LAB RESULTS heading.
        lab_y = None
        next_section_y = page_image.height
        for i, word in enumerate(words):
            if word["lower"] in ("lab", "laboratory"):
                nearby = words[i + 1:i + 6]
                if any(
                    item["lower"] in ("results", "result")
                    and abs(item["y"] - word["y"]) < 100
                    for item in nearby
                ):
                    lab_y = (
                        word["y"]
                        if lab_y is None
                        else min(lab_y, word["y"])
                    )
        if lab_y is None:
            print("[OCR] Lab heading not found; using full-page OCR only.")
            return ""
# Stop before the next section.
        for word in words:
            if word["lower"] in (
                "medications",
                "medication",
                "vitals",
                "observations",
                "conclusion",
            ) and word["y"] > lab_y:
                next_section_y = min(next_section_y, word["y"])
        crop_top = min(
            page_image.height - 1,
            lab_y + max(25, int(page_image.height * 0.015)),
        )
        crop_bottom = max(
            crop_top + 1,
            next_section_y - max(8, int(page_image.height * 0.006)),
        )
        section = page_image.crop(
            (0, crop_top, page_image.width, crop_bottom)
        )
        gray = cv2.cvtColor(
            np.asarray(section),
            cv2.COLOR_RGB2GRAY,
        )
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            9,
        )
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(25, section.width // 24), 1),
        )
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(15, section.height // 12)),
        )
        horizontal = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, h_kernel
        )
        vertical = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, v_kernel
        )
        def grouped_positions(mask, axis, minimum_count):
            projection = np.count_nonzero(mask, axis=axis)
            positions = np.where(
                projection >= minimum_count
            )[0].tolist()
            groups = []
            for pos in positions:
                if not groups or pos > groups[-1][-1] + 2:
                    groups.append([pos])
                else:
                    groups[-1].append(pos)
            return [
                int(sum(group) / len(group))
                for group in groups
            ]
        horizontal_ys = grouped_positions(
            horizontal,
            axis=1,
            minimum_count=int(section.width * 0.22),
        )
        horizontal_ys = [
            y for y in horizontal_ys
            if 0 <= y < section.height
        ]
        if len(horizontal_ys) >= 3:
            best_group = []
            current = [horizontal_ys[0]]
            for y in horizontal_ys[1:]:
                if y - current[-1] < section.height * 0.22:
                    current.append(y)
                else:
                    if len(current) > len(best_group):
                        best_group = current
                    current = [y]
            if len(current) > len(best_group):
                best_group = current
            table_ys = best_group
            if len(table_ys) >= 3:
                top_y, bottom_y = table_ys[0], table_ys[-1]
                table_vertical = vertical[top_y:bottom_y + 1, :]
                vertical_xs = grouped_positions(
                    table_vertical,
                    axis=0,
                    minimum_count=max(
                        6,
                        int((bottom_y - top_y) * 0.12),
                    ),
                )
                vertical_xs = [
                    x for x in vertical_xs
                    if 0 <= x < section.width
                ]
                if len(vertical_xs) >= 4:
                    if len(vertical_xs) > 5:
                        vertical_xs = sorted(
                            vertical_xs,
                            key=lambda x: sum(
                                vertical[
                                    :,
                                    max(0, x - 2):
                                    min(vertical.shape[1], x + 3)
                                ].sum(axis=0)
                            ),
                            reverse=True,
                        )[:5]
                        vertical_xs = sorted(vertical_xs)
                    x_bounds = vertical_xs
                    y_bounds = table_ys
                    output = ["LAB RESULTS - ROW-WISE OCR"]
                    detected_header = None
                    for row_idx in range(len(y_bounds) - 1):
                        y1 = y_bounds[row_idx] + 2
                        y2 = y_bounds[row_idx + 1] - 2
                        if y2 <= y1:
                            continue
                        cells = []
                        for col_idx in range(len(x_bounds) - 1):
                            x1 = x_bounds[col_idx] + 2
                            x2 = x_bounds[col_idx + 1] - 2
                            if x2 <= x1:
                                cells.append("")
                                continue
                            cell = section.crop((x1, y1, x2, y2))
                            cell = ImageOps.grayscale(cell)
                            cell = ImageOps.autocontrast(cell)
                            cell = ImageOps.expand(
                                cell, border=8, fill="white"
                            )
                            cell = cell.resize(
                                (cell.width * 3, cell.height * 3),
                                Image.Resampling.LANCZOS,
                            )
                            try:
                                cell_text = pytesseract.image_to_string(
                                    cell,
                                    lang="eng",
                                    config="--oem 1 --psm 7",
                                    timeout=8,
                                ).strip()
                            except Exception as exc:
                                print(
                                    f"[WARN] Lab cell OCR failed: {exc}"
                                )
                                cell_text = ""
                            cells.append(" ".join(cell_text.split()))
                        # Detect the header from OCR text instead of assuming
                        # that every table has exactly four columns.
                        if row_idx == 0:
                            normalized_cells = [
                                re.sub(r"[^a-z]", "", value.casefold())
                                for value in cells
                            ]
                            is_header = (
                                any("test" in value for value in normalized_cells)
                                and any("name" in value for value in normalized_cells)
                                and any("result" in value for value in normalized_cells)
                            )
                            if is_header:
                                detected_header = []
                                for value in cells:
                                    low = re.sub(r"[^a-z]", "", value.casefold())
                                    if "test" in low and "name" in low:
                                        detected_header.append("Test Name")
                                    elif "result" in low:
                                        detected_header.append("Result")
                                    elif "unit" in low:
                                        detected_header.append("Unit")
                                    elif "reference" in low or "range" in low:
                                        detected_header.append("Reference Range")
                                    elif "flag" in low or "status" in low:
                                        detected_header.append("Flag")
                                    else:
                                        detected_header.append(value or "Other")
                                output.append("\\t".join(detected_header))
                                continue

                        if detected_header is None and row_idx == 0:
                            output.append("\\t".join(
                                f"Column {col + 1}" for col in range(len(cells))
                            ))
                        if any(cells):
                            output.append("\\t".join(cells))
                    if len(output) > 2:
                        print(
                            "[OCR] Grid-based lab table OCR: "
                            f"{len(output) - 2} data rows, "
                            f"{len(x_bounds) - 1} columns."
                        )
                        return "\\n".join(output)

# Fallback: use approximate column positions inside the lab section.
        try:
            left = int(section.width * 0.055)
            right = int(section.width * 0.94)
            table = section.crop((left, 0, right, section.height))
            gray_table = ImageOps.autocontrast(
                ImageOps.grayscale(table)
            )
            table_data = pytesseract.image_to_data(
                gray_table,
                lang="eng",
                config="--oem 1 --psm 6",
                output_type=pytesseract.Output.DICT,
                timeout=15,
            )
            header_y = None
            for i, raw in enumerate(table_data["text"]):
                token = (raw or "").strip().casefold().strip(":.,")
                if token in (
                    "test", "name", "result", "unit", "reference", "range"
                ):
                    y = int(table_data["top"][i])
                    if header_y is None or y < header_y:
                        header_y = y
            if header_y is not None:
                x_edges = [
                    0,
                    int(table.width * 0.39),
                    int(table.width * 0.59),
                    int(table.width * 0.73),
                    table.width,
                ]
                gray_np = np.asarray(gray_table)
                dark = gray_np < 150
                row_projection = dark.sum(axis=1)
                line_candidates = np.where(
                    row_projection > table.width * 0.30
                )[0].tolist()
                line_groups = []
                for yy in line_candidates:
                    if (
                        not line_groups
                        or yy > line_groups[-1][-1] + 2
                    ):
                        line_groups.append([yy])
                    else:
                        line_groups[-1].append(yy)
                rule_ys = [
                    int(sum(group) / len(group))
                    for group in line_groups
                ]
                rule_ys = [
                    y for y in rule_ys if y > header_y + 8
                ]
                if len(rule_ys) >= 3:
                    row_edges = [
                        max(header_y + 8, rule_ys[0])
                    ] + rule_ys[1:]
                    fallback_rows = []
                    for ri in range(len(row_edges) - 1):
                        y1 = row_edges[ri] + 2
                        y2 = row_edges[ri + 1] - 2
                        if y2 - y1 < 8:
                            continue
                        cells = []
                        for ci in range(4):
                            cell = gray_table.crop((
                                x_edges[ci] + 2,
                                y1,
                                x_edges[ci + 1] - 2,
                                y2,
                            ))
                            cell = ImageOps.expand(
                                cell, border=6, fill="white"
                            )
                            cell = cell.resize(
                                (
                                    cell.width * 3,
                                    cell.height * 3,
                                ),
                                Image.Resampling.LANCZOS,
                            )
                            txt = pytesseract.image_to_string(
                                cell,
                                lang="eng",
                                config="--oem 1 --psm 7",
                                timeout=8,
                            )
                            cells.append(" ".join(txt.split()))
                        if any(cells) and not any(
                            value.casefold() in (
                                "test name",
                                "result",
                                "unit",
                                "reference range",
                            )
                            for value in cells
                        ):
                            fallback_rows.append("\t".join(cells))
                    if fallback_rows:
                        print(
                            "[OCR] Proportional lab-cell fallback: "
                            f"{len(fallback_rows)} rows."
                        )
                        return (
                            "LAB RESULTS - ROW-WISE OCR\n"
                            "Test Name\tResult\tUnit\tReference Range\n"
                            + "\n".join(fallback_rows)
                        )
        except Exception as exc:
            print(
                f"[WARN] Proportional lab-cell fallback failed: {exc}"
            )
# Last fallback: OCR the entire lab section.
        gray_image = ImageOps.autocontrast(
            ImageOps.grayscale(section)
        )
        candidates = []
        for variant_name, variant in (
            ("gray", gray_image),
            ("rgb", section),
        ):
            for psm in (6, 4, 11):
                try:
                    text = pytesseract.image_to_string(
                        variant,
                        lang="eng",
                        config=f"--oem 1 --psm {psm}",
                        timeout=15,
                    ).strip()
                    if text:
                        candidates.append(
                            (
                                _text_quality(text),
                                text,
                                variant_name,
                                psm,
                            )
                        )
                except Exception as exc:
                    print(f"[WARN] Table crop OCR failed: {exc}")
        if candidates:
            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )
            print(
                f"[OCR] Grid detection fallback={candidates[0][2]}, "
                f"PSM={candidates[0][3]}"
            )
            return candidates[0][1]
        return ""
    except Exception as exc:
        print(f"[WARN] Separate lab-table OCR failed: {exc}")
        return ""
def _needs_second_engine(text):
    if not text or len(text.strip()) < 80:
        return True
    lower = text.casefold()
    has_report_fields = any(
        term in lower
        for term in ("patient", "hospital", "doctor", "report")
    )
    has_lab_section = (
        "lab results" in lower or "laboratory" in lower
    )
    if has_lab_section:
        position = max(
            lower.find("lab results"),
            lower.find("laboratory"),
        )
        lab_chunk = lower[position:position + 700]
        if sum(ch.isalnum() for ch in lab_chunk) < 45:
            return True
    return not has_report_fields
def extract_printed_text(page: dict) -> dict:
    """Extract printed text from a PDF page or scanned image."""
    total_start = time.perf_counter()
    page_number = page.get("page", 1)
    embedded_text = page.get("text", "")
    if not isinstance(embedded_text, str):
        embedded_text = ""
    embedded_text = embedded_text.strip()
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
            error="No image or embedded text was provided.",
        )
    try:
        start = time.perf_counter()
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise ValueError("Image preprocessing returned no image.")
        print(
            f"[TIMING] Page {page_number} preprocessing: "
            f"{time.perf_counter() - start:.2f}s"
        )
        start = time.perf_counter()
        try:
            recognized_text = _run_tesseract(processed_image)
        except Exception as exc:
            recognized_text = ""
            print(f"[WARN] Tesseract failed: {exc}")
        print(
            f"[TIMING] Page {page_number} Tesseract: "
            f"{time.perf_counter() - start:.2f}s"
        )
        engine = "tesseract"
        if _needs_second_engine(recognized_text):
            print(
                f"[OCR] Page {page_number}: trying PaddleOCR fallback."
            )
            paddle_text = _run_paddleocr(processed_image)
            if (
                paddle_text
                and _text_quality(paddle_text)
                > _text_quality(recognized_text)
            ):
                recognized_text = paddle_text
                engine = "paddleocr"
# Run the separate table OCR against the original page image.
        lab_table_text = _extract_lab_table_text(image)
        final_text = recognized_text or embedded_text
        if lab_table_text:
            final_text = (
                final_text
                + "\n\nLAB TABLE ROW OCR (separate crop):\n"
                + lab_table_text
            ).strip()
        if not final_text:
            return _result(
                page,
                engine,
                "no_text_detected",
                error="No readable text detected by OCR.",
            )
        return _result(
            page,
            engine,
            "text_extracted",
            final_text,
        )
    except Exception as exc:
        print(
            f"[ERROR] Page {page_number}: "
            f"{type(exc).__name__}: {exc}"
        )
        return _result(
            page,
            "tesseract",
            "error",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        print(
            f"[TIMING] Page {page_number} total OCR time: "
            f"{time.perf_counter() - total_start:.2f}s"
        )
