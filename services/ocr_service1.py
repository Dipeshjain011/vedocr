
import time

from services.printed_ocr import extract_printed_text
from services.field_extractor import extract_fields, infer_document_type
from services.name_corrector import correct_name
from services.medical_extractor import extract_medical_data

from services.metadata_extractor import extract_document_metadata
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
    """Remove internal helper fields and normalize extracted names."""
    fields = dict(fields)

    internal_fields = {
        "document_type_from_document",
        "document_date",
        "document_date_original",
        "report_date_original",
        "hospital_name_raw",
        "admission_date_original",
        "discharge_date_original",
    }

    for key in internal_fields:
        fields.pop(key, None)

    patient_name = fields.get("patient_name")

    if isinstance(patient_name, str) and patient_name.strip():
        fields["patient_name"] = correct_name(patient_name)

    # Keep only populated fields. Preserve valid values such as "Not specified"
    # if the field extractor explicitly returned them.
    return {
        key: value
        for key, value in fields.items()
        if value is not None
    }


def _empty_medical_data() -> dict:
    """Return a consistent response when medical extraction is unavailable."""
    return {
        "lab_results": [],
        "clinical_information": {},
        "medications": [],
        "vitals": {},
        "diagnostic_reports": {},
    }


def process_document(pages: list[dict]) -> dict:
    """Process document pages and prepare structured VedAmrit OCR output."""
    document_start = time.perf_counter()

    page_results = []
    warnings = []

    # 1. Extract text from every page.
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
                f"Page {page_number}: text extraction was unsuccessful "
                f"({detail})."
            )

    # 2. Combine text from all pages in their original order.
    raw_text = "\n\n".join(
        result.get("text", "").strip()
        for result in page_results
        if isinstance(result.get("text"), str)
        and result.get("text", "").strip()
    ).strip()

    # 3. Extract patient and document metadata.
    if raw_text:
        try:
            fields, field_warnings = extract_fields(raw_text)

            if not isinstance(fields, dict):
                fields = {}
                warnings.append(
                    "Document field extraction returned an unexpected format."
                )

            warnings.extend(field_warnings or [])
            fields = _clean_fields(fields)

            document_type = infer_document_type(raw_text)

        except Exception as exc:
            fields = {}
            document_type = "medical_report"

            warnings.append(
                "Patient/document field extraction failed; "
                "the remaining OCR text was preserved."
            )

            print(
                "[OCR] Field extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

    else:
        fields = {}
        document_type = "medical_report"

        warnings.append(
            "No readable text was extracted. "
            "Structured fields could not be populated."
        )

    # 4. Extract lab results, vitals, medicines, and clinical information.
    if raw_text:
        try:
            medical_data = extract_medical_data(raw_text)

            if not isinstance(medical_data, dict):
                raise TypeError(
                    "extract_medical_data() must return a dictionary."
                )

            defaults = _empty_medical_data()

            for key, default_value in defaults.items():
                if not isinstance(medical_data.get(key), type(default_value)):
                    medical_data[key] = default_value

        except Exception as exc:
            medical_data = _empty_medical_data()

            warnings.append(
                "Medical data extraction failed. "
                "The extracted document fields and raw text were preserved."
            )

            print(
                "[OCR] Medical extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

    else:
        medical_data = _empty_medical_data()

    # 5. Do not mark OCR output as medically verified automatically.
    review_required = True
    verified = False

    if raw_text and not medical_data["lab_results"]:
        warnings.append(
            "No structured laboratory results were detected. "
            "Check the original document and raw OCR text."
        )

    # 6. Final response.
    elapsed = time.perf_counter() - document_start

    print(
        f"[OCR] Document processing finished in {elapsed:.2f}s "
        f"across {len(page_results)} page(s)."
    )

    return {
        "document_type": document_type,
        "fields": fields,
        "lab_results": medical_data["lab_results"],
        "clinical_information": medical_data["clinical_information"],
        "medications": medical_data["medications"],
        "vitals": medical_data["vitals"],
        "diagnostic_reports": medical_data["diagnostic_reports"],
        "pages": page_results,
        "raw_text": raw_text,
        "review_required": review_required,
        "verified": verified,
        "warnings": list(dict.fromkeys(warnings)),
    }
