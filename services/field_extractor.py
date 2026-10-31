import re
from datetime import datetime
from typing import Optional
# ==================================================
# FIELD LABELS
# ==================================================
LABELS = {
    "patient_name": [
        r"Patient\s+name",
        r"Name\s+of\s+patient",
        r"Patient\s+full\s+name",
        r"Name",
    ],
    "doctor_name": [
        r"Doctor\s+name",
        r"Consultant\s+doctor",
        r"Referring\s+doctor",
        r"Attending\s+doctor",
        r"Physician\s+name",
        r"Ref\s+By",
        r"Referred\s+By",
    ],
    "hospital_name": [
        r"Hospital\s+name",
        r"Hospital\s*/\s*Lab",
        r"Hospital\s*/\s*Laboratory",
        r"Laboratory\s+name",
        r"Lab\s+name",
        r"Diagnostic\s+centre",
        r"Diagnostic\s+center",
        r"Clinic\s+name",
    ],
    "admission_date": [
        r"Admission\s+date",
        r"Admit\s+date",
        r"Date\s+of\s+admission",
    ],
    "discharge_date": [
        r"Discharge\s+date",
        r"Date\s+of\s+discharge",
    ],
    "report_date": [
        r"Report\s+date",
        r"Date\s+of\s+report",
        r"Reporting\s+date",
        r"Date\s+reported",
        r"Investigation\s+date",
        r"Date\s+of\s+investigation",
        r"Test\s+date",
        r"Date\s+of\s+test",
        r"Sample\s+collection\s+date",
        r"Date\s+of\s+collection",
        r"Specimen\s+collection\s+date",
        r"Collected",
        r"Reported",
    ],
    "report_id": [
        r"Report\s+ID",
        r"Report\s+No\.?",
        r"Report\s+Number",
        r"Lab\s+Report\s+ID",
        r"Test\s+ID",
        r"Accession\s+Number",
        r"Specimen\s+ID",
    ],
    "patient_id": [
        r"Patient\s+ID",
        r"Patient\s+No\\\.?",
        r"Patient\s+Number",
        r"Medical\s+Record\s+Number",
        r"MRN",
        r"UHID",
    ],
    "lab_id": [
        r"Lab\s+ID",
        r"Laboratory\s+ID",
        r"Laboratory\s+Number",
    ],
    "document_type": [
        r"Document\s+type",
    ],
}
DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",  # Supports dates such as 2026/10/21*
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d-%m-%y",
    "%d/%m/%y",
    "%d %b %Y",
    "%d %B %Y",
    "%d %b %y",
    "%d %B %y",
]
# ==================================================
# CLEANING HELPERS
# ==================================================
def clean_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = re.sub(r"\s+", " ", value.strip())
    value = value.strip(" \t:;|")
    if not value:
        return None
    if value.casefold() in {
        "n/a",
        "na",
        "none",
        "null",
        "not available",
        "not provided",
        "-",
    }:
        return None
    return value
def _label_patterns():
    return [
        pattern
        for patterns in LABELS.values()
        for pattern in patterns
    ]
def matches_label(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    for pattern in _label_patterns():
        if re.fullmatch(
            r"(?:" + pattern + r")\s*:?",
            line,
            re.IGNORECASE,
        ):
            return True
    return False
def is_section_boundary(line: str) -> bool:
    if not line:
        return True
    normalized = re.sub(
        r"[\s:|]+", " ", line
    ).strip().casefold()
    headings = {
        "field",
        "details",
        "patient and report information",
        "patient information",
        "patient details",
        "report information",
        "clinical information",
        "clinical notes",
        "medical history",
        "diagnostic report",
        "diagnostic reports",
        "investigation report",
        "laboratory report",
        "lab report",
        "laboratory investigations",
        "lab investigations",
        "test results",
        "laboratory results",
        "investigation results",
        "vital signs",
        "vitals",
        "medications",
        "medication",
        "prescription",
        "treatment and procedures",
        "conclusion",
        "recommendations",
        "impression",
        "doctor's notes",
        "doctors notes",
    }
    if normalized in headings:
        return True
    if re.match(
        r"^\d+[\\\.)]\s+[A-Za-z][A-Za-z /&()-]{2,}$",
        line.strip(),
    ):
        return True
    return False
def _is_usable_value(line: str) -> bool:
    return bool(
        line
        and not matches_label(line)
        and not is_section_boundary(line)
    )
# ==================================================
# FIELD VALUE EXTRACTION
# ==================================================
def find_field_value(
    text: str,
    patterns: list[str],
) -> Optional[str]:
    """Extract a field from inline or separate-line OCR text.
    Handles lines such as \`\`Patient Name: Aarav Sharma\`\` and trims a value
    if another known field label was accidentally joined onto the same line.
    """
    lines = [line.strip() for line in text.splitlines()]
    # Recognize any known label so OCR-merged fields can be separated.*
    all_labels = _label_patterns()
    next_label_pattern = re.compile(
        r"\s+(?=(?:" + "|".join(all_labels) + r")\s*[:：])",
        re.IGNORECASE,
    )
    # First, look for labels and values on the same line.*
    for pattern in patterns:
        regex = re.compile(
            r"(?<!\w)(?:" + pattern + r")"
            r"\s*[:：-]\s*(.*?)\s*$",
            re.IGNORECASE,
        )
        for line in lines:
            match = regex.search(line)
            if not match:
                continue
            value = clean_value(match.group(1))
            if not value:
                continue
            value = clean_value(next_label_pattern.split(value, maxsplit=1)[0])
            if value and _is_usable_value(value):
                return value
    # Then, look for labels on their own line and take the next usable value.*
    for pattern in patterns:
        label_regex = re.compile(
            r"(?:" + pattern + r")\s*[:：]?",
            re.IGNORECASE,
        )
        for index, line in enumerate(lines):
            if not label_regex.fullmatch(line):
                continue
            for candidate in lines[index + 1:index + 4]:
                candidate = clean_value(candidate)
                if not candidate:
                    continue
                if not _is_usable_value(candidate):
                    continue
                # Do not accept a line that starts with another field label.*
                candidate = clean_value(next_label_pattern.split(candidate, maxsplit=1)[0])
                if candidate and _is_usable_value(candidate):
                    return candidate
    return None
# ==================================================
# ID EXTRACTION
# ==================================================
def find_id_value(
    text: str,
    patterns: list[str],
) -> Optional[str]:
    """
    Extract identifiers from inline or multiline labels.
    Example:
        Patient ID: DEMO-PAT-2046
    Or:
        Patient ID
        DEMO-PAT-2046
    """
    lines = [line.strip() for line in text.splitlines()]
    id_pattern = r"([A-Za-z0-9][A-Za-z0-9/_-]*)"
    # Inline identifier.*
    for pattern in patterns:
        regex = re.compile(
            r"(?<!\w)(?:" + pattern + r")"
            r"\s*[:\\\\-]\s*" + id_pattern,
            re.IGNORECASE,
        )
        for line in lines:
            match = regex.search(line)
            if match:
                return clean_value(match.group(1))
    # Identifier on a following line.*
    for pattern in patterns:
        label_regex = re.compile(
            r"(?:" + pattern + r")\s*:?",
            re.IGNORECASE,
        )
        for index, line in enumerate(lines):
            if not label_regex.fullmatch(line):
                continue
            for candidate in lines[index + 1:index + 5]:
                candidate = clean_value(candidate)
                if not candidate:
                    continue
                if candidate.casefold() in {
                    "field",
                    "details",
                    "patient and report information",
                }:
                    continue
                if matches_label(candidate):
                    continue
                if is_section_boundary(candidate):
                    continue
                match = re.match(
                    r"^([A-Za-z0-9][A-Za-z0-9/_-]*)",
                    candidate,
                )
                if match:
                    return match.group(1)
    return None
# ==================================================
# DATE NORMALIZATION
# ==================================================
def normalize_date(
    value: Optional[str],
) -> Optional[str]:
    """Normalize recognizable dates to YYYY-MM-DD."""
    value = clean_value(value)
    if not value:
        return None
    value = re.sub(
        r"^(date|admission date|discharge date)\s*[:-]?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(\d{1,2})(st|nd|rd|th)\b",
        r"\1",
        value,
        flags=re.IGNORECASE,
    )
    value = value.strip(" .,:;")
    for date_format in DATE_FORMATS:
        try:
            parsed = datetime.strptime(value, date_format)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Try to find a date inside a longer value.*
    date_match = re.search(
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b"
        r"|\b\d{4}-\d{1,2}-\d{1,2}\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        value,
    )
    if date_match:
        extracted = date_match.group(0)
        for date_format in DATE_FORMATS:
            try:
                parsed = datetime.strptime(
                    extracted, date_format
                )
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
    return value
def extract_labeled_date(
    text: str,
    patterns: list[str],
) -> Optional[str]:
    value = find_field_value(text, patterns)
    if not value:
        return None
    return normalize_date(value)
# ==================================================
# HOSPITAL EXTRACTION
# ==================================================
def extract_hospital_raw(
    text: str,
) -> Optional[str]:
    value = find_field_value(
        text,
        LABELS["hospital_name"],
    )
    value = clean_value(value)
    if not value:
        return None
    if matches_label(value) or is_section_boundary(value):
        return None
    return value
# ==================================================
# EXPLICIT DOCUMENT TYPE
# ==================================================
def extract_document_type(
    text: str,
) -> Optional[str]:
    value = find_field_value(
        text,
        LABELS["document_type"],
    )
    return clean_value(value)
# ==================================================
# DOCUMENT TYPE CLASSIFICATION
# ==================================================
def infer_document_type(
    text: str,
    filename: str = "",
) -> str:
    """
    Rule-based classification.
    Laboratory evidence is checked before prescription
    keywords because a lab report may contain a
    prescription template or medication section.
    """
    combined = f"{filename}\n{text}".lower()
    # 1. Explicit document type, if available.*
    explicit_type = extract_document_type(text)
    if explicit_type:
        value = explicit_type.lower()
        if any(term in value for term in [
            "lab",
            "laboratory",
            "pathology",
        ]):
            return "lab_report"
        if "prescription" in value or "medication" in value:
            return "prescription"
        if "discharge" in value:
            return "discharge_summary"
        if any(term in value for term in [
            "radiology",
            "imaging",
            "x-ray",
            "xray",
            "mri",
            "ct scan",
        ]):
            return "radiology_report"
        if "medical report" in value or "clinical report" in value:
            return "medical_report"
    # 2. Laboratory report detection.*
    lab_terms = [
        "comprehensive sample laboratory report",
        "laboratory report",
        "lab report",
        "complete blood count",
        "reference range",
        "laboratory investigations",
        "lab investigations",
        "test results",
        "hemoglobin",
        "haemoglobin",
        "blood glucose",
        "serum creatinine",
        "urinalysis",
        "urine analysis",
    ]
    lab_score = sum(
        term in combined for term in lab_terms
    )
    if lab_score >= 2:
        return "lab_report"
    # 3. Radiology report detection.*
    radiology_terms = [
        "radiology report",
        "ultrasound report",
        "x-ray report",
        "xray report",
        "mri report",
        "ct scan report",
        "radiological impression",
        "imaging findings",
    ]
    if any(term in combined for term in radiology_terms):
        return "radiology_report"
    # 4. Discharge summary detection.*
    discharge_terms = [
        "discharge summary",
        "discharge diagnosis",
        "condition at discharge",
        "discharge instructions",
    ]
    if sum(
        term in combined for term in discharge_terms
    ) >= 2:
        return "discharge_summary"
    # 5. Prescription detection.*
    prescription_terms = [
        "prescription",
        "prescribed medicines",
        "medication instructions",
        "take one tablet",
        "take 1 tablet",
        "twice daily",
        "once daily",
        "dosage",
    ]
    if sum(
        term in combined for term in prescription_terms
    ) >= 2:
        return "prescription"
    # 6. General medical report detection.*
    medical_terms = [
        "medical report",
        "clinical information",
        "medical history",
        "clinical findings",
        "patient history",
        "diagnostic report",
    ]
    # A report heading alone is enough to classify a general medical report.*
    if any(term in combined for term in (
        "medical report",
        "clinical report",
    )):
        return "medical_report"
    if sum(term in combined for term in medical_terms) >= 2:
        return "medical_report"
    return "unknown"
# ==================================================
# MAIN FIELD EXTRACTION
# ==================================================
def extract_fields(text: str):
    """
    Extract structured fields.
    Returns:
        (fields, warnings)
    This two-value return is compatible with:
        fields, field_warnings = extract_fields(raw_text)
    """
    fields = {}
    warnings = []
    if not text or not text.strip():
        warnings.append(
            "No text available for field extraction."
        )
        return fields, warnings
    # Patient name.*
    value = find_field_value(
        text, LABELS["patient_name"]
    )
    if value:
        fields["patient_name"] = clean_value(value)
    # Doctor name.*
    value = find_field_value(
        text, LABELS["doctor_name"]
    )
    if value:
        fields["doctor_name"] = clean_value(value)
    # Hospital name.*
    hospital_name = extract_hospital_raw(text)
    if hospital_name:
        fields["hospital_name"] = hospital_name
        fields["hospital_name_raw"] = hospital_name
    # Admission date.*
    value = find_field_value(
        text, LABELS["admission_date"]
    )
    if value:
        fields["admission_date_original"] = value
        fields["admission_date"] = normalize_date(value)
    # Discharge date.*
    value = find_field_value(
        text, LABELS["discharge_date"]
    )
    if value:
        fields["discharge_date_original"] = value
        fields["discharge_date"] = normalize_date(value)
    # Report date: prefer the actual date printed on the report.
    value = find_field_value(text, LABELS["report_date"])
    if value:
        normalized_report_date = normalize_date(value)
        if normalized_report_date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized_report_date):
            fields["report_date"] = normalized_report_date
            fields["report_date_original"] = value

    # Common laboratory-report layout: "Name : Ms. ... UHID ...",
    # "Ref By : ...", and timestamp fields "Collected" / "Reported".
    # These fallbacks are deliberately label-specific to avoid guessing from arbitrary text.
    if not fields.get("patient_name"):
        match = re.search(r"(?im)^\s*Name\s*[:：]\s*(.+?)(?=\s+UHID\b|\s+Lab\s+No\b|$)", text)
        if match:
            candidate = clean_value(match.group(1))
            if candidate and not re.search(r"\b(?:UHID|Lab\s+No|Age|Gender)\b", candidate, re.I):
                fields["patient_name"] = re.sub(r"^(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+", "", candidate, flags=re.I).strip()
    if not fields.get("doctor_name"):
        match = re.search(r"(?im)^\s*(?:Ref\s+By|Referred\s+By)\s*[:：]\s*([^\n]+)", text)
        if match:
            candidate = clean_value(match.group(1))
            if candidate:
                fields["doctor_name"] = candidate
                fields["referring_doctor"] = candidate
    # Prefer the report's explicit "Reported" timestamp for report_date;
    # otherwise use the "Collected" timestamp as the specimen date.
    if not fields.get("report_date"):
        match = re.search(r"(?im)^.*?\bReported\s*[:：]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?:\s+([^\n]+))?", text)
        if match:
            original = match.group(1)
            normalized = normalize_date(original)
            if normalized and re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
                fields["report_date"] = normalized
                fields["report_date_original"] = original
    if not fields.get("collection_date"):
        match = re.search(r"(?im)^.*?\bCollected\s*[:：]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
        if match:
            original = match.group(1)
            normalized = normalize_date(original)
            if normalized and re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
                fields["collection_date"] = normalized
                fields["collection_date_original"] = original
    # Normalize common OCR header rows that combine multiple fields on one line.
    # Example: "Name : Ms. KUSUM MENARIA UHID 59856 ..."
    match = re.search(r"(?im)^\s*Name\s*[:：]\s*(.+?)(?=\s+UHID\b|\s+Lab\s+No\b|\s+Age\s*[:：]|\s+Gender\s*[:：]|$)", text)
    if match:
        candidate = clean_value(match.group(1))
        if candidate:
            fields["patient_name"] = re.sub(r"^(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+", "", candidate, flags=re.I).strip()
    match = re.search(r"(?im)\b(?:Ref\s+By|Referred\s+By)\s*[:：]\s*(.+?)(?=\s+Gender\s*[:：]|\s+Collected\s*[:：]|$)", text)
    if match:
        candidate = clean_value(match.group(1))
        if candidate:
            fields["doctor_name"] = candidate
            fields["referring_doctor"] = candidate
    # For lab reports, "Reported" is the report-generation timestamp, while
    # "Collected" is the specimen collection timestamp. Prioritize these labels.
    match = re.search(r"(?im)\bReported\s*[:：]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
    if match:
        original = match.group(1)
        normalized = normalize_date(original)
        if normalized and re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
            fields["report_date"] = normalized
            fields["report_date_original"] = original
    match = re.search(r"(?im)\bCollected\s*[:：]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
    if match:
        original = match.group(1)
        normalized = normalize_date(original)
        if normalized and re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
            fields["collection_date"] = normalized
            fields["collection_date_original"] = original
    # Report ID.*
    value = find_id_value(

        text, LABELS["report_id"]
    )
    if value:
        fields["report_id"] = value
    # Patient ID.*
    value = find_id_value(
        text, LABELS["patient_id"]
    )
    if value:
        fields["patient_id"] = value
    # Laboratory ID.*
    value = find_id_value(
        text, LABELS["lab_id"]
    )
    if value:
        fields["lab_id"] = value
    # Explicit document type, if present.*
    value = extract_document_type(text)
    if value:
        fields["document_type_from_document"] = value
    # Remove empty values.*
    fields = {
        key: value
        for key, value in fields.items()
        if value is not None and value != ""
    }
    if not fields:
        warnings.append(
            "No structured patient or report fields were extracted."
        )
    return fields, warnings
# ==================================================
# ADDITIONAL STRUCTURED SECTIONS
# ==================================================
CLINICAL_LABELS = {
    "reason_for_visit": [r"Reason\s+for\s+visit", r"Chief\s+complaint"],
    "medical_condition": [r"Medical\s+condition", r"Current\s+condition", r"Diagnosis"],
    "allergy_note": [r"Allergy\s+note", r"Allergies?"],
}
def _section_lines(text: str, heading_patterns: list[str], stop_headings: set[str]) -> list[str]:
    """Return lines after a section heading until the next known section."""
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    start = None
    heading_re = re.compile(r"^(?:" + "|".join(heading_patterns) + r")\s*:?\s*$", re.I)
    for i, line in enumerate(lines):
        if heading_re.fullmatch(line):
            start = i + 1
            break
    if start is None:
        return []
    result = []
    for line in lines[start:]:
        normalized = line.strip(" :").casefold()
        if normalized in stop_headings:
            break
        if line:
            result.append(line)
    return result
def extract_clinical_information(text: str) -> dict:
    """Extract labeled clinical context without inventing missing values."""
    result = {}
    for key, patterns in CLINICAL_LABELS.items():
        value = find_field_value(text, patterns)
        if value:
            result[key] = value
    return result
def extract_medications(text: str) -> list[dict]:
    """Extract medication entries while skipping table headers and orphan values."""
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    start = None
    for i, line in enumerate(lines):
        if re.fullmatch(r"MEDICATIONS?|PRESCRIPTION", line, re.I):
            start = i + 1
            break
    if start is None:
        return []
    stop = {"vital signs", "vitals", "diagnostic reports", "diagnostic report",
            "clinical information", "laboratory results", "lab results"}
    section = []
    for line in lines[start:]:
        if line.casefold().strip(" :") in stop:
            break
        if line:
            section.append(line)
    # Exclude column labels and obvious table fragments.*
    ignored = {"medicine name and instructions", "medicine", "medicine name",
               "medication", "strength", "frequency", "dose", "duration",
               "instructions"}
    section = [line for line in section if line.casefold().strip(" :") not in ignored]
    entries = []
    for line in section:
        # The main medication description usually contains a name and instruction.*
        is_orphan_strength = bool(re.fullmatch(r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)", line, re.I))
        is_orphan_frequency = bool(re.fullmatch(
            r"(?:once daily|twice daily|thrice daily|once a day|twice a day|daily|at night|as needed)",
            line, re.I
        ))
        if (not is_orphan_strength and not is_orphan_frequency
                and re.search(r"\b(mg|mcg|g|ml|tablet|capsule|once daily|twice daily|daily)\b", line, re.I)):
            if not any(line.casefold() == e["medicine_name_and_instructions"].casefold()
                       for e in entries):
                item = {"medicine_name_and_instructions": line}
                strength = re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)\b", line, re.I)
                frequency = re.search(r"\b(once daily|twice daily|thrice daily|once a day|twice a day|daily|at night|as needed)\b", line, re.I)
                if strength:
                    item["strength"] = strength.group(0)
                if frequency:
                    item["frequency"] = frequency.group(0)
                entries.append(item)
    return entries
def extract_diagnostic_reports(text: str) -> list[dict]:
    """Extract diagnostic table rows; never treat the column headers as reports."""
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    start = None
    for i, line in enumerate(lines):
        if re.fullmatch(r"DIAGNOSTIC REPORTS?", line, re.I):
            start = i + 1
            break
    if start is None:
        return []
    stop = {"ocr test note", "vital signs", "vitals", "medications",
            "clinical information", "laboratory results", "lab results"}
    section = []
    for line in lines[start:]:
        low = line.casefold().strip(" :")
        if any(low.startswith(x) for x in stop):
            break
        if line:
            section.append(line)
    headers = {"report type", "finding / result", "report date", "count"}
    section = [line for line in section if line.casefold().strip(" :") not in headers]
    reports = []
    # Support OCR text that places each table cell on its own line:*
    # type, finding, date, repeated for each row.*
    date_re = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
    i = 0
    while i < len(section):
        report_type = section[i]
        if report_type.casefold().strip(" :") in headers:
            i += 1
            continue
        finding = section[i + 1] if i + 1 < len(section) else ""
        report_date = section[i + 2] if i + 2 < len(section) else ""
        if finding and (date_re.fullmatch(report_date) or i + 1 < len(section)):
            row = {"report_type": report_type, "finding": finding}
            if date_re.fullmatch(report_date):
                row["report_date"] = normalize_date(report_date)
                i += 3
            else:
                i += 2
            # Avoid accidentally accepting section headings as rows.*
            if report_type.casefold() not in headers and not is_section_boundary(report_type):
                reports.append(row)
        else:
            i += 1
    return reports
def extract_structured_sections(text: str) -> dict:
    """Convenience wrapper for the three additional API response sections."""
    return {
        "clinical_information": extract_clinical_information(text),
        "medications": extract_medications(text),
        "diagnostic_reports": extract_diagnostic_reports(text),
    }
