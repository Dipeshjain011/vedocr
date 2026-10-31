
import time

from services.printed_ocr import extract_printed_text
from services.field_extractor import extract_fields, infer_document_type
from services.name_corrector import correct_name
from services.medical_extractor import extract_medical_data
from services.metadata_extractor import extract_document_metadata


# REPORT DATE HAS THE HIGHEST PRIORITY.

TIMELINE_PRIORITY = [
    ("report_date", "high"),
    ("document_date", "high"),
    ("sample_collection_date", "medium"),
    ("admission_date", "medium"),
    ("order_date", "medium"),
    ("discharge_date", "medium"),
]


def process_page(page: dict, page_number: int) -> dict:

    """Extract readable text from a single document page."""

    start_time = time.perf_counter()

    print(f"[OCR] Page {page_number}: processing started")

    try:

        result = extract_printed_text(page)

        if not isinstance(result, dict):

            raise TypeError(
                "extract_printed_text() must return a dictionary."
            )

        result["page"] = page_number

        result.setdefault("engine", "unknown")
        result.setdefault("status", "success")
        result.setdefault("text", "")

        return result

    finally:

        elapsed = time.perf_counter() - start_time

        print(
            f"[OCR] Page {page_number}: "
            f"processing finished in {elapsed:.2f}s"
        )


def _clean_fields(fields: dict) -> dict:

    """Remove internal fields and normalize patient name."""

    fields = dict(fields)

    internal_fields = {
    "document_type_from_document",
    "hospital_name_raw",
    "report_date_original",
    "document_date_original",
    "sample_collection_date_original",
    "admission_date_original",
    "order_date_original",
    "discharge_date_original",
    "patient_id",
    "pathologist_name",
}

    for key in internal_fields:
        fields.pop(key, None)

    patient_name = fields.get("patient_name")

    if isinstance(patient_name, str) and patient_name.strip():

        fields["patient_name"] = correct_name(patient_name)

    return {
        key: value
        for key, value in fields.items()
        if value is not None and value != ""
    }


def _empty_medical_data() -> dict:

    """Return consistent empty medical data."""

    return {
        "lab_results": [],
        "clinical_information": {},
        "medications": [],
        "vitals": {},
        "diagnostic_reports": {},
    }


def _empty_metadata() -> dict:

    """Default metadata when OCR cannot read the document."""

    return {
        "document_type": "unknown",
        "fields": {},
        "timeline": {
            "date": None,
            "date_source": None,
            "confidence": None,
        },
    }


def _choose_timeline(
    fields: dict,
    metadata_timeline: dict | None = None
) -> dict:

    """
    Select timeline date using fixed priority.

    1. Report Date
    2. Document Date
    3. Sample Collection Date
    4. Admission Date
    5. Order Date
    6. Discharge Date
    """

    for field, confidence in TIMELINE_PRIORITY:

        value = fields.get(field)

        if value:

            return {
                "date": value,
                "date_source": field,
                "confidence": confidence,
            }

    # Do not use a timeline date that is not supported
    # by the extracted date fields.

    return {
        "date": None,
        "date_source": None,
        "confidence": None,
    }


def process_document(
    pages: list[dict],
    filename: str = ""
) -> dict:

    """
    Process all medical document pages.

    Main purpose:
    - Extract patient identity.
    - Extract hospital and doctor details.
    - Extract all available document dates.
    - Select Report Date as the primary timeline date.
    - Use fallback dates only if Report Date is missing.
    - Preserve existing clinical extraction.
    """

    document_start = time.perf_counter()

    page_results = []
    warnings = []

    # --------------------------------------------------
    # 1. OCR TEXT EXTRACTION
    # --------------------------------------------------

    for page_number, page in enumerate(pages, start=1):

        try:

            result = process_page(page, page_number)

        except Exception as exc:

            result = {
                "page": page_number,
                "engine": "unknown",
                "status": "error",
                "text": "",
                "error": f"{type(exc).__name__}: {exc}",
            }

            print(
                f"[OCR] Page {page_number} failed: "
                f"{type(exc).__name__}: {exc}"
            )

        page_results.append(result)

        if result.get("status") in {
            "error",
            "setup_required",
            "no_text_detected",
        }:

            detail = result.get("error") or result.get("status")

            warnings.append(
                f"Page {page_number}: text extraction unsuccessful "
                f"({detail})."
            )

    # --------------------------------------------------
    # 2. COMBINE TEXT FROM ALL PAGES
    # --------------------------------------------------

    raw_text = "\n\n".join(
        result.get("text", "").strip()
        for result in page_results
        if isinstance(result.get("text"), str)
        and result.get("text", "").strip()
    ).strip()

    fields = {}

    metadata = _empty_metadata()

    document_type = "unknown"

    # --------------------------------------------------
    # 3. EXTRACT COMMON MEDICAL DOCUMENT METADATA
    # --------------------------------------------------

    if raw_text:

        try:

            metadata = extract_document_metadata(
                raw_text,
                filename
            )

            if not isinstance(metadata, dict):

                raise TypeError(
                    "Metadata extractor must return a dictionary."
                )

            metadata_fields = metadata.get("fields", {})

            if isinstance(metadata_fields, dict):

                fields.update(metadata_fields)

            document_type = metadata.get(
                "document_type",
                "unknown"
            )

        except Exception as exc:

            warnings.append(
                "Common medical metadata extraction failed."
            )

            print(
                "[OCR] Metadata extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

        # --------------------------------------------------
        # 4. EXISTING FIELD EXTRACTOR AS FALLBACK
        # --------------------------------------------------

        try:

            existing_fields, field_warnings = extract_fields(
                raw_text
            )

            if isinstance(existing_fields, dict):

                for key, value in existing_fields.items():

                    if (
                        value is not None
                        and value != ""
                        and not fields.get(key)
                    ):

                        fields[key] = value

            warnings.extend(field_warnings or [])

        except Exception as exc:

            warnings.append(
                "Additional field extraction failed. "
                "Metadata extraction results were preserved."
            )

            print(
                "[OCR] Existing field extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

        # --------------------------------------------------
        # 5. DOCUMENT TYPE FALLBACK
        # --------------------------------------------------

        if not document_type or document_type == "unknown":

            try:

                document_type = infer_document_type(raw_text)

            except Exception as exc:

                document_type = "unknown"

                print(
                    "[OCR] Document classification failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        if not document_type:
            document_type = "unknown"

        # --------------------------------------------------
        # 6. CLEAN EXTRACTED FIELDS
        # --------------------------------------------------

        fields = _clean_fields(fields)

        # --------------------------------------------------
        # 7. TIMELINE DATE
        # --------------------------------------------------

        timeline = _choose_timeline(
            fields,
            metadata.get("timeline", {})
        )

        if not timeline["date"]:

            warnings.append(
                "No reliable document date was detected. "
                "Manual date verification is required."
            )

        if not fields.get("patient_name"):
            warnings.append("Patient name not detected.")

        if not fields.get("hospital_name"):
            warnings.append(
                "Hospital or laboratory name not detected."
            )

        if not fields.get("doctor_name"):
            warnings.append(
                "Doctor or consultant name not detected."
            )

    else:

        warnings.append(
            "No readable text was extracted. "
            "Metadata could not be populated."
        )

        timeline = _empty_metadata()["timeline"]

    # --------------------------------------------------
    # 8. OPTIONAL MEDICAL INFORMATION EXTRACTION
    # --------------------------------------------------

    if raw_text:

        try:

            medical_data = extract_medical_data(raw_text)

            if not isinstance(medical_data, dict):

                raise TypeError(
                    "Medical extractor must return a dictionary."
                )

            defaults = _empty_medical_data()

            for key, default_value in defaults.items():

                if not isinstance(
                    medical_data.get(key),
                    type(default_value)
                ):

                    medical_data[key] = default_value

        except Exception as exc:

            medical_data = _empty_medical_data()

            warnings.append(
                "Optional clinical information extraction failed. "
                "Document metadata and OCR text were preserved."
            )

            print(
                "[OCR] Medical extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

    else:

        medical_data = _empty_medical_data()

    # --------------------------------------------------
    # 9. VERIFICATION STATUS
    # --------------------------------------------------

    review_required = True

    verified = False

    # --------------------------------------------------
    # 10. FINAL RESPONSE
    # --------------------------------------------------

    elapsed = time.perf_counter() - document_start

    print(
        f"[OCR] Document processing finished in {elapsed:.2f}s "
        f"across {len(page_results)} page(s)."
    )

    return {

        "filename": filename,

        "document_type": document_type,

        "fields": fields,

        "timeline": timeline,

        # Optional clinical extraction

        "lab_results": medical_data["lab_results"],

        "clinical_information": medical_data["clinical_information"],

        "medications": medical_data["medications"],

        "vitals": medical_data["vitals"],

        "diagnostic_reports": medical_data["diagnostic_reports"],

        # OCR information

        "pages": page_results,

        "raw_text": raw_text,

        # Verification

        "review_required": review_required,

        "verified": verified,

        "warnings": list(dict.fromkeys(warnings)),

    }
