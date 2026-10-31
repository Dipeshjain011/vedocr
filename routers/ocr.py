import json
import re
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from config import MAX_UPLOAD_MB
from schemas import OCRPage, OCRResponse
from services.document_loader import load_upload_bytes
from services.ocr_service import process_document
from services.field_extractor import extract_structured_sections
from services.lab_results_extractor import extract_lab_results
from services.pdf_report import create_medical_report_pdf


router = APIRouter(tags=["Medical Reports OCR"])


def _result_date(item: dict, sort_by: str) -> str | None:
    """Get a normalized date from the selected field or timeline."""
    timeline = item.get("timeline") or {}
    fields = item.get("fields") or {}

    if sort_by == "timeline_date":
        value = timeline.get("date")
    else:
        value = fields.get(sort_by)

    if not value and sort_by != "timeline_date":
        value = timeline.get("date")

    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return None


def _sort_ocr_results(results: list[dict], sort_by: str, order: str) -> list[dict]:
    """Sort dated reports chronologically; keep undated reports at the end."""
    dated = []
    undated = []

    for index, item in enumerate(results):
        date_value = _result_date(item, sort_by)
        if date_value:
            dated.append((date_value, index, item))
        else:
            undated.append((index, item))

    dated.sort(key=lambda row: row[0], reverse=(order == "desc"))
    return [row[2] for row in dated] + [row[1] for row in undated]

ALLOWED = {".pdf", ".png", ".jpg", ".jpeg"}
VALID_TYPES = {
    "auto",
    "lab_report",
    "medical_report",
    "prescription",
    "discharge_summary",
    "admission_record",
    "radiology_report",
    "medical_bill",
    "medical_certificate",
    "unknown",
}

SORT_FIELDS = {
    "timeline_date",
    "report_date",
    "document_date",
    "sample_collection_date",
    "order_date",
    "admission_date",
    "discharge_date",
}

ALLOWED_FIELD_KEYS = {
    "patient_name",
    "hospital_name",
    "doctor_name",
    "document_date",
    "order_date",
    "sample_collection_date",
    "report_date",
    "document_time",
    "order_time",
    "sample_collection_time",
    "report_time",
    "admission_date",
    "discharge_date",
    "report_id",
    "lab_id",
}


def _merge_extracted_data(
    existing: Any,
    extracted: Any,
    default: Any,
) -> Any:
    """Keep existing non-empty data; use extracted data as fallback."""

    if existing is None:
        existing = default

    if isinstance(existing, dict):
        return existing if existing else extracted

    if isinstance(existing, list):
        return existing if existing else extracted

    return existing


def _normalize_diagnostic_reports(value: Any) -> dict:
    """Ensure diagnostic_reports is always a dictionary."""

    if isinstance(value, dict):
        return value

    if isinstance(value, list):
        return {"reports": value} if value else {}

    return {}


async def _process(
    file: UploadFile,
    document_type: str,
) -> OCRResponse:
    """Process one uploaded PDF or image."""

    request_start = time.perf_counter()

    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()

    print(f"[API] Started processing: {filename}")

    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail="Upload a PDF, PNG, JPG, or JPEG file.",
        )

    if document_type not in VALID_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "document_type must be auto or a supported medical document type."
            ),
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
        )

    try:
        load_start = time.perf_counter()
        pages = load_upload_bytes(content, suffix)
        load_elapsed = time.perf_counter() - load_start

        print(
            f"[API] Document loaded in {load_elapsed:.2f}s; "
            f"pages: {len(pages)}"
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No pages or images could be loaded from the file.",
        )

    try:
        result = process_document(
            pages,
            filename=filename
        )

    except Exception as exc:
        print(f"[API] OCR processing failed: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Document processing failed. Check the server logs.",
        ) from exc

    raw_text = result.get("raw_text", "") or ""
    fields = result.get("fields", {}) or {}
    timeline = result.get("timeline", {}) or {}
    warnings = list(result.get("warnings", []) or [])

    if document_type == "auto":
        resolved_type = result.get("document_type") or "medical_report"
    else:
        resolved_type = document_type

    fields = {
        key: value
        for key, value in fields.items()
        if key in ALLOWED_FIELD_KEYS and value is not None
    }

    lab_results = result.get("lab_results") or []
    clinical_information = result.get("clinical_information") or {}
    medications = result.get("medications") or []
    vitals = result.get("vitals") or {}

    diagnostic_reports = _normalize_diagnostic_reports(
        result.get("diagnostic_reports")
    )

    if not lab_results and raw_text.strip():
        try:
            lab_results = extract_lab_results(raw_text)

            print(
                f"[API] Lab results extracted: {len(lab_results)}"
            )

            if not lab_results:
                warnings.append(
                    "No structured lab results were detected "
                    "from the extracted text."
                )

        except Exception as exc:
            print(f"[API] Lab extraction failed: {exc}")

            warnings.append(
                "Lab test results could not be structured automatically."
            )

    try:
        extracted_sections = extract_structured_sections(raw_text) or {}

        clinical_information = _merge_extracted_data(
            clinical_information,
            extracted_sections.get("clinical_information", {}),
            {},
        )

        medications = _merge_extracted_data(
            medications,
            extracted_sections.get("medications", []),
            [],
        )

        extracted_diagnostics = _normalize_diagnostic_reports(
            extracted_sections.get("diagnostic_reports", {})
        )

        diagnostic_reports = _merge_extracted_data(
            diagnostic_reports,
            extracted_diagnostics,
            {},
        )

    except Exception as exc:
        print(f"[API] Additional extraction warning: {exc}")

        warnings.append(
            "Additional medical-section extraction was unsuccessful."
        )

    if not raw_text.strip():
        warnings.append(
            "No readable text was extracted. "
            "Structured fields could not be populated."
        )

    if not fields.get("patient_name"):
        warnings.append(
            "Patient name was not confidently detected."
        )

    warnings.append(
        "OCR fields are unverified. Check names and dates "
        "against the original document."
    )

    warnings = list(dict.fromkeys(warnings))

    try:
        validated_pages = [
            OCRPage(**page_result)
            for page_result in result.get("pages", [])
        ]

    except Exception as exc:
        print(f"[API] Page validation failed: {exc}")

        raise HTTPException(
            status_code=500,
            detail="OCR page output did not match the response schema.",
        ) from exc

    elapsed = time.perf_counter() - request_start

    print(
        f"[API] Finished processing {filename} "
        f"in {elapsed:.2f}s"
    )

    return OCRResponse(
        filename=filename,
        document_type=resolved_type,
        fields=fields,
        timeline=timeline,
        lab_results=lab_results,
        clinical_information=clinical_information,
        medications=medications,
        vitals=vitals,
        diagnostic_reports=diagnostic_reports,
        pages=validated_pages,
        raw_text=raw_text,
        review_required=True,
        verified=False,
        warnings=warnings,
    )


@router.post(
    "/ocr",
    response_model=OCRResponse,
    response_model_exclude_none=True,
)
async def ocr_one(
    file: UploadFile = File(...),
    document_type: str = Form("auto"),
):
    """OCR a single medical or laboratory report."""

    return await _process(file, document_type)


@router.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    document_type: str = Form("auto"),
):
    """Process one document and return a downloadable PDF report."""

    ocr_response = await _process(file, document_type)
    report_data = ocr_response.model_dump(exclude_none=True)

    try:
        pdf_bytes = create_medical_report_pdf(report_data)

    except Exception as exc:
        print(f"[API] PDF generation failed: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Could not generate the medical report PDF.",
        ) from exc

    source_name = Path(ocr_response.filename).stem

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        source_name,
    ).strip("_")

    safe_name = safe_name or "medical_report"
    download_name = f"{safe_name}_vedamrit_report.pdf"

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{download_name}"'
            )
        },
    )


@router.post("/ocr/batch")
async def ocr_batch(
    files: List[UploadFile] = File(
        ...,
        description="Upload multiple PDF, PNG, JPG, or JPEG reports",
    ),
    document_type: str = Form("auto"),
    sort_by: str = Form("timeline_date"),
    order: str = Form("asc"),
):
    """OCR multiple reports and return date-sorted JSON results."""

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one document.",
        )

    if document_type not in VALID_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "document_type must be auto or a supported medical document type."
            ),
        )

    if sort_by not in SORT_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=(
                "sort_by must be one of: " + ", ".join(sorted(SORT_FIELDS))
            ),
        )

    if order not in {"asc", "desc"}:
        raise HTTPException(
            status_code=400,
            detail="order must be asc or desc.",
        )

    results = []

    for uploaded_file in files:
        result = await _process(uploaded_file, document_type)
        results.append(result.model_dump(exclude_none=True))

    sorted_results = _sort_ocr_results(results, sort_by, order)

    return {
        "count": len(sorted_results),
        "sort_by": sort_by,
        "order": order,
        "date_format": "YYYY-MM-DD",
        "results": sorted_results,
        "review_required": True,
    }


@router.post("/ocr/batch/pdf-zip")
async def ocr_batch_pdf_zip(
    files: List[UploadFile] = File(
        ...,
        description="Upload multiple PDF, PNG, JPG, or JPEG reports",
    ),
    document_type: str = Form("auto"),
    sort_by: str = Form("timeline_date"),
    order: str = Form("asc"),
):
    """
    Process multiple reports and return one ZIP containing:
      - original_files/: exact uploaded source files
      - extracted_json/: one OCR JSON per uploaded file
      - all_results.json: combined chronologically sorted results

    Date priority: report_date, document_date, sample_collection_date,
    admission_date, order_date, discharge_date.
    Undated and failed documents are placed at the end.

    sort_by is retained for compatibility with the previous endpoint.
    The ZIP always uses the VedAmrit date-priority sequence above.
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one document.",
        )

    if document_type not in VALID_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "document_type must be auto or a supported medical document type."
            ),
        )

    if sort_by not in SORT_FIELDS:
        raise HTTPException(
            status_code=400,
            detail="sort_by must be one of: " + ", ".join(sorted(SORT_FIELDS)),
        )

    if order not in {"asc", "desc"}:
        raise HTTPException(
            status_code=400,
            detail="order must be asc or desc.",
        )

    date_priority = [
        "report_date",
        "document_date",
        "sample_collection_date",
        "admission_date",
        "order_date",
        "discharge_date",
    ]

    def get_priority_date(report_data: dict) -> str | None:
        fields = report_data.get("fields") or {}
        for field_name in date_priority:
            value = fields.get(field_name)
            if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                return value

        # Fallback only when no prioritized field is populated.
        timeline = report_data.get("timeline") or {}
        value = timeline.get("date")
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return value
        return None

    original_files = []
    processed_results = []

    # Read every upload once, preserve its original bytes, then pass a fresh
    # in-memory UploadFile to the existing _process() OCR pipeline.
    for index, uploaded_file in enumerate(files, start=1):
        original_filename = Path(uploaded_file.filename or f"upload_{index}").name
        extension = Path(original_filename).suffix.lower()

        try:
            content = await uploaded_file.read()
        except Exception as exc:
            processed_results.append({
                "filename": original_filename,
                "status": "failed",
                "error": f"Could not read uploaded file: {exc}",
                "review_required": True,
                "verified": False,
            })
            continue
        finally:
            await uploaded_file.close()

        # Preserve the source upload in the archive, even if OCR validation fails.
        original_files.append({
            "filename": original_filename,
            "content": content,
        })

        if extension not in ALLOWED:
            processed_results.append({
                "filename": original_filename,
                "status": "failed",
                "error": "Unsupported file format. Upload PDF, PNG, JPG, or JPEG.",
                "review_required": True,
                "verified": False,
            })
            continue

        if not content:
            processed_results.append({
                "filename": original_filename,
                "status": "failed",
                "error": "Uploaded file is empty.",
                "review_required": True,
                "verified": False,
            })
            continue

        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            processed_results.append({
                "filename": original_filename,
                "status": "failed",
                "error": f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
                "review_required": True,
                "verified": False,
            })
            continue

        fresh_upload = UploadFile(
            filename=original_filename,
            file=BytesIO(content),
        )

        try:
            ocr_response = await _process(fresh_upload, document_type)
            report_data = ocr_response.model_dump(exclude_none=True)
            report_data["status"] = "success"
            processed_results.append(report_data)

        except HTTPException as exc:
            processed_results.append({
                "filename": original_filename,
                "status": "failed",
                "error": exc.detail,
                "http_status": exc.status_code,
                "review_required": True,
                "verified": False,
            })

        except Exception as exc:
            print(f"[API] Batch OCR failed for {original_filename}: {exc}")
            processed_results.append({
                "filename": original_filename,
                "status": "failed",
                "error": "Document processing failed. Check server logs.",
                "review_required": True,
                "verified": False,
            })

        finally:
            await fresh_upload.close()

    # Sort successful, dated reports first. Keep stable ordering for equal dates.
    successful = [r for r in processed_results if r.get("status") == "success"]
    failed = [r for r in processed_results if r.get("status") != "success"]
    dated = [r for r in successful if get_priority_date(r)]
    undated = [r for r in successful if not get_priority_date(r)]
    dated.sort(key=lambda r: get_priority_date(r), reverse=(order == "desc"))
    sorted_results = dated + undated + failed

    combined_data = {
        "project": "VedAmrit",
        "total_uploaded": len(files),
        "total_successful": len(successful),
        "total_failed": len(failed),
        "sorting": {
            "date_priority": date_priority,
            "order": order,
            "date_format": "YYYY-MM-DD",
            "undated_documents": "placed_at_end",
        },
        "results": sorted_results,
        "review_required": True,
        "verified": False,
    }

    zip_buffer = BytesIO()
    used_original_names = set()
    used_json_names = set()

    def make_unique_filename(filename: str, used_names: set) -> str:
        path = Path(filename)
        candidate = filename
        counter = 2
        while candidate.casefold() in used_names:
            candidate = f"{path.stem}_{counter}{path.suffix}"
            counter += 1
        used_names.add(candidate.casefold())
        return candidate

    try:
        with zipfile.ZipFile(
            zip_buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            # 1. Original uploaded files
            for item in original_files:
                safe_name = re.sub(
                    r"[^A-Za-z0-9._-]+", "_", item["filename"]
                ).strip("._") or "uploaded_file"
                safe_name = make_unique_filename(safe_name, used_original_names)
                archive.writestr(
                    f"original_files/{safe_name}",
                    item["content"],
                )

            # 2. One JSON file for every processed/failed upload
            for index, report_data in enumerate(sorted_results, start=1):
                source_name = Path(
                    report_data.get("filename") or f"document_{index}"
                ).stem
                safe_stem = re.sub(
                    r"[^A-Za-z0-9_-]+", "_", source_name
                ).strip("_") or f"document_{index}"
                json_name = make_unique_filename(
                    f"{index:03d}_{safe_stem}.json",
                    used_json_names,
                )
                archive.writestr(
                    f"extracted_json/{json_name}",
                    json.dumps(
                        report_data,
                        indent=4,
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8"),
                )

            # 3. Combined chronological JSON
            archive.writestr(
                "all_results.json",
                json.dumps(
                    combined_data,
                    indent=4,
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8"),
            )

    except Exception as exc:
        print(f"[API] Batch ZIP generation failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Could not generate the batch ZIP. Check server logs.",
        ) from exc

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                'attachment; filename="vedamrit_reports_sorted.zip"'
            ),
            "Content-Length": str(zip_buffer.getbuffer().nbytes),
        },
)