

import os

# Disable oneDNN/MKLDNN before PaddleOCR is imported.
os.environ["FLAGS_use_mkldnn"] = "0"

import time

import re

from functools import lru_cache



import numpy as np

import cv2

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



    # Avoid aggressively enlarging already-readable phone/scanner images.
    # A modest cap keeps small text legible without multiplying OCR pixels.
    if max(width, height) > 2000:
        scale = 2000 / max(width, height)
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

        "patient",

        "hospital",

        "doctor",

        "report",

        "lab results",

        "test name",

        "reference",

        "result",

        "hemoglobin",

        "glucose",

        "medications",

        "cetirizine",

        "vitals",

    ]



    score += sum(

        2.5 for word in keywords if word in lower

    )



    return score





def _run_tesseract(image):
    """Fast OCR: one primary pass, with at most two targeted retries."""
    get_tesseract_version()
    variants = _ocr_variants(image)
    if not variants:
        return ""

    first_name, first_image = variants[0]
    candidates = []

    # Primary pass: PSM 11 works well for mixed medical report layouts.
    try:
        text = pytesseract.image_to_string(
            first_image,
            lang="eng",
            config="--oem 1 --psm 11",
            timeout=12,
        ).strip()
        if text:
            score = _text_quality(text)
            candidates.append((score, text, first_name, 11))
            # Good readable text: do not spend time on other variants.
            if len(text) >= 55 and score >= 20:
                print(
                    f"[OCR] Fast path variant={first_name}, "
                    f"PSM=11, quality={score:.1f}"
                )
                return text
    except Exception as exc:
        print(f"[WARN] Tesseract primary pass failed: {exc}")

    # Targeted retry 1: same image with block-oriented segmentation.
    try:
        text = pytesseract.image_to_string(
            first_image,
            lang="eng",
            config="--oem 1 --psm 6",
            timeout=12,
        ).strip()
        if text:
            candidates.append((_text_quality(text), text, first_name, 6))
    except Exception as exc:
        print(f"[WARN] Tesseract PSM 6 fallback failed: {exc}")

    # Targeted retry 2: grayscale/contrast image, only when still weak.
    best = max(candidates, key=lambda item: item[0]) if candidates else None
    if (best is None or len(best[1]) < 55 or best[0] < 20) and len(variants) > 1:
        variant_name, variant = variants[1]
        try:
            text = pytesseract.image_to_string(
                variant,
                lang="eng",
                config="--oem 1 --psm 11",
                timeout=12,
            ).strip()
            if text:
                candidates.append((_text_quality(text), text, variant_name, 11))
        except Exception as exc:
            print(f"[WARN] Tesseract grayscale fallback failed: {exc}")

    if not candidates:
        return ""

    best = max(candidates, key=lambda item: item[0])
    print(
        f"[OCR] Best variant={best[2]}, PSM={best[3]}, "
        f"quality={best[0]:.1f}"
    )
    return best[1]

@lru_cache(maxsize=1)
def _get_paddleocr_engine():
    """Initialize the optional PaddleOCR model once per worker process."""
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _run_paddleocr(image):

    """Optional PaddleOCR fallback for difficult scanned pages."""



    try:

        ocr = _get_paddleocr_engine()



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

                    for key in (

                        "res",

                        "result",

                        "results",

                        "data",

                    ):

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



def _norm_header_word(s):

    return re.sub(r"[^a-z]", "", s.casefold())





def _group_ocr_lines(tokens):

    tokens = sorted(tokens, key=lambda t: (t["cy"], t["x"]))

    lines = []



    for token in tokens:

        target = None



        for line in reversed(lines[-5:]):

            cy = sum(t["cy"] for t in line) / len(line)

            h = sorted(t["h"] for t in line)[len(line) // 2]



            if abs(token["cy"] - cy) <= max(8, h * 0.9):

                target = line

                break



        if target is None:

            lines.append([token])

        else:

            target.append(token)



    for line in lines:

        line.sort(key=lambda t: t["x"])



    lines.sort(key=lambda line: min(t["cy"] for t in line))



    return lines





def _header_columns(header_line):

    test_words = [

        t for t in header_line

        if _norm_header_word(t["text"]) in {"test", "name"}

    ]



    result_words = [

        t for t in header_line

        if _norm_header_word(t["text"]) == "result"

    ]



    unit_words = [

        t for t in header_line

        if _norm_header_word(t["text"]) == "unit"

    ]



    ref_words = [

        t for t in header_line

        if _norm_header_word(t["text"]) in {"reference", "range"}

    ]



    flag_words = [

        t for t in header_line

        if _norm_header_word(t["text"]) == "flag"

    ]



    if not (test_words and result_words and ref_words):

        return None



    def avg(words_):

        return sum(

            t["x"] + t["w"] / 2 for t in words_

        ) / len(words_)



    cols = [

        ("test_name", avg(test_words)),

        ("result", avg(result_words)),

    ]



    if unit_words:

        cols.append(("unit", avg(unit_words)))



    cols.append(("reference_range", avg(ref_words)))



    if flag_words:

        cols.append(("flag", avg(flag_words)))



    cols.sort(key=lambda x: x[1])



    if any(

        cols[i][1] >= cols[i + 1][1]

        for i in range(len(cols) - 1)

    ):

        return None



    return cols





def _find_header(lines):

    for idx, line in enumerate(lines):

        cols = _header_columns(line)



        if cols and len(cols) >= 4:

            return idx, line, cols



    return None, None, None





def _read_lab_table_cells(image):

    """

    Read a printed lab table by detecting its grid and OCRing each cell.



    Table coordinates and cell contents are derived from the supplied image.

    Medical test names and result values are not hardcoded.

    """



    page = (

        image

        if isinstance(image, Image.Image)

        else Image.fromarray(image)

    )



    page = ImageOps.exif_transpose(page).convert("RGB")

    w, h = page.size



    scale = (

        min(2.0, 2600 / max(w, h))

        if max(w, h) < 2600

        else 1.0

    )



    if scale != 1.0:

        page = page.resize(

            (int(w * scale), int(h * scale)),

            Image.Resampling.LANCZOS,

        )



    rgb = np.asarray(page)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)



    binary = cv2.adaptiveThreshold(

        gray,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY_INV,

        31,

        12,

    )



    height, width = binary.shape



    # Detect table-header words and their coordinates.

    data = pytesseract.image_to_data(

        page,

        lang="eng",

        config="--oem 1 --psm 11",

        output_type=pytesseract.Output.DICT,

        timeout=20,

    )



    tokens = []



    for i, raw in enumerate(data["text"]):

        text = (raw or "").strip()



        if not text:

            continue



        try:

            conf = float(data["conf"][i])

        except (ValueError, TypeError):

            conf = 0



        if conf < 5:

            continue



        x = int(data["left"][i])

        y = int(data["top"][i])

        ww = int(data["width"][i])

        hh = int(data["height"][i])



        tokens.append({

            "text": text,

            "x": x,

            "y": y,

            "w": ww,

            "h": hh,

            "cy": y + hh / 2,

            "conf": conf,

        })



    if not tokens:

        return ""



    lines = _group_ocr_lines(tokens)

    header_idx, header_line, columns = _find_header(lines)



    if header_line is None:

        print("[OCR] Lab table header not detected.")

        return ""



    header_y = sum(

        t["cy"] for t in header_line

    ) / len(header_line)



    # Detect horizontal and vertical table lines.

    horizontal_kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (max(30, width // 35), 1),

    )



    vertical_kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (1, max(25, height // 45)),

    )



    horizontal = cv2.morphologyEx(

        binary, cv2.MORPH_OPEN, horizontal_kernel

    )



    vertical = cv2.morphologyEx(

        binary, cv2.MORPH_OPEN, vertical_kernel

    )



    def grouped_positions(values, max_gap=3):

        groups = []



        for value in values:

            if (

                not groups

                or value > groups[-1][-1] + max_gap

            ):

                groups.append([int(value)])

            else:

                groups[-1].append(int(value))



        return [

            int(round(sum(group) / len(group)))

            for group in groups

        ]



    horizontal_strength = np.sum(horizontal > 0, axis=1)



    h_positions = grouped_positions(

        np.where(horizontal_strength > width * 0.14)[0],

        3,

    )



    if len(h_positions) < 5:

        print("[OCR] Not enough table row rules detected.")

        return ""



    above = [

        y for y in h_positions

        if y <= header_y + 2

    ]



    if not above:

        return ""



    top_rule = max(above)



    table_h_rules = [

        y for y in h_positions

        if top_rule <= y <= top_rule + height * 0.28

    ]



    if len(table_h_rules) < 3:

        return ""



    # Keep the table grid closest to the detected header.

    selected_h = [table_h_rules[0]]



    for y in table_h_rules[1:]:

        gap = y - selected_h[-1]



        if gap > max(120, height * 0.055):

            break



        selected_h.append(y)



    if len(selected_h) < 3:

        return ""



    top, bottom = selected_h[0], selected_h[-1]



    vertical_strength = np.sum(

        vertical[top:bottom + 1] > 0,

        axis=0,

    )



    x_positions = grouped_positions(

        np.where(

            vertical_strength > (bottom - top) * 0.55

        )[0],

        3,

    )



    x_positions = [

        x for x in x_positions

        if 0 <= x < width

    ]



    if len(x_positions) < 4:

        print("[OCR] Not enough table column rules detected.")

        return ""



    # Ignore vertical lines near page borders.

    x_positions = [

        x for x in x_positions

        if width * 0.04 <= x <= width * 0.95

    ]



    if len(x_positions) < 4:

        return ""



    # OCR each table cell separately.

    rows = []



    for row_index in range(len(selected_h) - 1):

        y1 = selected_h[row_index]

        y2 = selected_h[row_index + 1]



        row = []



        for col_index in range(len(x_positions) - 1):

            x1 = x_positions[col_index]

            x2 = x_positions[col_index + 1]



            pad_x = max(2, int((x2 - x1) * 0.012))

            pad_y = max(2, int((y2 - y1) * 0.08))



            cell = gray[

                max(0, y1 + pad_y):min(height, y2 - pad_y),

                max(0, x1 + pad_x):min(width, x2 - pad_x),

            ]



            if cell.size == 0:

                row.append("")

                continue



            cell = cv2.resize(

                cell,

                None,

                fx=2,

                fy=2,

                interpolation=cv2.INTER_CUBIC,

            )



            cell = cv2.copyMakeBorder(

                cell,

                8,

                8,

                8,

                8,

                cv2.BORDER_CONSTANT,

                value=255,

            )



            text = pytesseract.image_to_string(

                cell,

                lang="eng",

                config="--oem 1 --psm 7",

                timeout=10,

            ).strip()



            text = re.sub(r"\s+", " ", text)

            text = re.sub(r"^[|\\\s]+|[|\\\s]+$", "", text)



            row.append(text)



        rows.append(row)



    if not rows:

        return ""



    # Confirm that the first detected row resembles a table header.

    header_cells = [

        re.sub(r"[^a-z]", "", value.casefold())

        for value in rows[0]

    ]



    def header_name(value):

        if "test" in value or value == "name":

            return "test_name"

        if "result" in value or value == "value":

            return "result"

        if "unit" in value:

            return "unit"

        if "reference" in value or "range" in value:

            return "reference_range"

        if value == "flag":

            return "flag"

        return "other"



    mapped_headers = [

        header_name(value) for value in header_cells

    ]



    if (

        not any(v == "test_name" for v in mapped_headers)

        or not any(v == "result" for v in mapped_headers)

    ):

        print(

            "[OCR] Detected grid but could not verify its header cells."

        )

        return ""



    # Re-read result cells to improve decimal-point recognition.

    result_col = next(

        (

            i for i, name in enumerate(mapped_headers)

            if name == "result"

        ),

        None,

    )



    output_rows = []



    for row_index, row in enumerate(rows[1:], start=1):



        if not row or not row[0].strip():

            continue



        if not any(cell.strip() for cell in row[1:]):

            continue



        if (

            result_col is not None

            and result_col < len(x_positions) - 1

        ):

            y1 = selected_h[row_index]

            y2 = selected_h[row_index + 1]

            x1 = x_positions[result_col]

            x2 = x_positions[result_col + 1]



            pad_x = max(2, int((x2 - x1) * 0.012))

            pad_y = max(2, int((y2 - y1) * 0.08))



            result_cell = gray[

                max(0, y1 + pad_y):min(height, y2 - pad_y),

                max(0, x1 + pad_x):min(width, x2 - pad_x),

            ]



            if result_cell.size:

                result_cell = cv2.threshold(

                    result_cell,

                    0,

                    255,

                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,

                )[1]



                result_cell = cv2.resize(

                    result_cell,

                    None,

                    fx=4,

                    fy=4,

                    interpolation=cv2.INTER_CUBIC,

                )



                result_cell = cv2.copyMakeBorder(

                    result_cell,

                    20,

                    20,

                    20,

                    20,

                    cv2.BORDER_CONSTANT,

                    value=255,

                )



                numeric_read = pytesseract.image_to_string(

                    result_cell,

                    lang="eng",

                    config=(

                        "--oem 1 --psm 7 "

                        "-c tessedit_char_whitelist=0123456789.,<>+-"

                    ),

                    timeout=10,

                ).strip()



                numeric_read = re.sub(r"\s+", "", numeric_read)



                if (

                    numeric_read

                    and re.fullmatch(

                        r"[<>]?[-+]?\d[\d,.]*",

                        numeric_read,

                    )

                ):

                    row[result_col] = numeric_read



        output_rows.append(row)



    if not output_rows:

        return ""



    header_text = "\t".join(

        (

            mapped_headers[i]

            if mapped_headers[i] != "other"

            else f"column_{i + 1}"

        )

        for i in range(len(mapped_headers))

    )



    return (

        "LAB RESULTS - OCR CELL READING\n"

        + header_text

        + "\n"

        + "\n".join(

            "\t".join(row) for row in output_rows

        )

    )





def _extract_lab_table_text(image):

    """

    Extract lab table rows using OCR word coordinates and detected headers.



    Column positions and cell contents are derived from the page image.

    """



    try:

        return _read_lab_table_cells(image)



    except Exception as exc:

        print(

            f"[WARN] Coordinate-based lab table OCR failed: {exc}"

        )

        return ""



def _needs_second_engine(text):
    """Avoid expensive second-engine calls when Tesseract produced usable text."""
    if not text or len(text.strip()) < 45:
        return True
    cleaned = text.strip()
    alnum_count = sum(ch.isalnum() for ch in cleaned)
    alnum_ratio = alnum_count / max(len(cleaned), 1)
    # Short/noisy OCR is the only automatic trigger. Missing a particular
    # keyword alone is not enough to launch another full OCR engine.
    return alnum_ratio < 0.42

def extract_printed_text(page: dict) -> dict:

    """Extract printed text from a PDF page or scanned image."""



    total_start = time.perf_counter()

    page_number = page.get("page", 1)



    embedded_text = page.get("text", "")



    if not isinstance(embedded_text, str):

        embedded_text = ""



    embedded_text = embedded_text.strip()



    # Use embedded PDF text directly when available.

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

        # Step 1: preprocess the image.

        start = time.perf_counter()



        processed_image = preprocess_image(image)



        if processed_image is None:

            raise ValueError(

                "Image preprocessing returned no image."

            )



        print(

            f"[TIMING] Page {page_number} preprocessing: "

            f"{time.perf_counter() - start:.2f}s"

        )



        # Step 2: extract text using Tesseract.

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



        # Step 3: use PaddleOCR only if the first OCR result needs help.

        if _needs_second_engine(recognized_text):

            print(

                f"[OCR] Page {page_number}: "

                "trying PaddleOCR fallback."

            )



            paddle_text = _run_paddleocr(processed_image)



            if (

                paddle_text

                and _text_quality(paddle_text)

                > _text_quality(recognized_text)

            ):

                recognized_text = paddle_text

                engine = "paddleocr"



        # Step 4: skip expensive cell-by-cell OCR when primary OCR already
        # captured a useful set of lab names and numeric values.
        lab_table_text = ""
        table_start = time.perf_counter()
        text_for_table_check = (recognized_text or "").casefold()

        lab_markers = {
            "haemoglobin", "hemoglobin", "rbc", "pcv", "mcv", "mch",
            "mchc", "rdw", "wbc", "lymphocytes", "monocytes",
            "neutrophils", "eosinophils", "basophils", "platelets",
        }
        found_markers = {
            marker for marker in lab_markers
            if re.search(r"\b" + re.escape(marker) + r"\b", text_for_table_check)
        }
        numeric_values = re.findall(r"\b\d+(?:\.\d+)?\b", text_for_table_check)

        # Require several distinct CBC analytes plus numeric readings.
        # If primary OCR has them, another full-page/cell OCR pass is redundant.
        primary_lab_text_is_usable = (
            len(found_markers) >= 4 and len(numeric_values) >= 8
        )

        if primary_lab_text_is_usable:
            print(
                f"[OCR] Page {page_number}: skipping extra table OCR; "
                f"primary OCR captured {len(found_markers)} lab markers "
                f"and {len(numeric_values)} numeric values."
            )
        else:
            has_test_header = bool(
                re.search(r"\b(test\s*name|parameter)\b", text_for_table_check)
            )
            has_result_header = bool(
                re.search(r"\b(results?|values?)\b", text_for_table_check)
            )
            has_reference_header = bool(
                re.search(
                    r"\b(reference\s*range|ref\s*range|normal\s*range|biological\s*ref\s*range)\b",
                    text_for_table_check,
                )
            )

            if has_test_header and has_result_header and has_reference_header:
                lab_table_text = _extract_lab_table_text(image)
            else:
                print(
                    f"[OCR] Page {page_number}: skipping extra table OCR; "
                    "complete lab-table headers not detected."
                )

        print(
            f"[TIMING] Page {page_number} table OCR decision/processing: "
            f"{time.perf_counter() - table_start:.2f}s"
        )

        # Step 5: combine the regular OCR text with table OCR.

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
