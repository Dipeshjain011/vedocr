
import re
from datetime import datetime
from typing import Optional


# =========================================================
# DATE AND TIME PATTERNS
# =========================================================

DATE_REGEX = re.compile(
    r"\b("
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"
    r")\b",
    re.IGNORECASE,
)

TIME_REGEX = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b",
    re.IGNORECASE,
)

DATE_FORMATS = [
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",
    "%d %b %y",
    "%d %B %y",
]


# =========================================================
# COMMON CLEANING FUNCTIONS
# =========================================================

def clean(value: Optional[str]) -> Optional[str]:

    if not value:
        return None

    value = re.sub(r"\s+", " ", value)

    value = value.strip(" \t:;|,<>-")

    if not value:
        return None

    if value.lower() in {
        "n/a",
        "na",
        "none",
        "null",
        "-",
        "not available",
    }:
        return None

    return value


def normalize_date(value: Optional[str]) -> Optional[str]:

    if not value:
        return None

    match = DATE_REGEX.search(value)

    if not match:
        return None

    raw_date = match.group(1).strip().replace(".", "/")

    for fmt in DATE_FORMATS:

        try:

            parsed = datetime.strptime(raw_date, fmt)

            if "%y" in fmt and parsed.year < 1970:
                parsed = parsed.replace(
                    year=parsed.year + 100
                )

            if 1900 <= parsed.year <= datetime.now().year + 2:

                return parsed.strftime("%Y-%m-%d")

        except ValueError:
            continue

    return None


# =========================================================
# GENERIC LABEL VALUE EXTRACTION
# =========================================================

def extract_labeled_value(
    text: str,
    labels: list[str]
) -> Optional[str]:

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    stop_labels = re.compile(
        r"^(Patient Name|Patient ID|UHID|Age|Gender|"
        r"Report Date|Sample Collected On|Sample Collection|"
        r"Admission Date|Discharge Date|Order Date|"
        r"Registration Date|Referring Doctor|"
        r"Primary Consultant|Report Status|Lab No\.?)\s*[:：]?$",
        re.IGNORECASE,
    )

    for label in labels:

        pattern = re.compile(
            rf"(?<!\w)(?:{label})\s*[:：-]?\s*(.*)$",
            re.IGNORECASE,
        )

        for index, line in enumerate(lines):

            match = pattern.search(line)

            if not match:
                continue

            value = clean(match.group(1))

            if value:
                return value

            # Check the next line only if it is not another label.

            if index + 1 < len(lines):

                next_line = clean(lines[index + 1])

                if next_line and not stop_labels.match(next_line):
                    return next_line

    return None


# =========================================================
# DATE EXTRACTION
# =========================================================

def extract_labeled_date(
    text: str,
    labels: list[str]
) -> tuple[Optional[str], Optional[str], Optional[str]]:

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for label in labels:

        pattern = re.compile(
            rf"(?<!\w)(?:{label})\s*[:：-]?\s*(.*)$",
            re.IGNORECASE,
        )

        for index, line in enumerate(lines):

            # Do not mistake Previous Visit Date for Document Date.

            if re.search(
                r"\bPrevious\s+Visit\s+Date\b",
                line,
                re.IGNORECASE,
            ):
                continue

            match = pattern.search(line)

            if not match:
                continue

            value = clean(match.group(1))

            # Date may be on the following OCR line.

            if not value and index + 1 < len(lines):
                value = clean(lines[index + 1])

            if not value:
                continue

            date_match = DATE_REGEX.search(value)

            if not date_match:
                continue

            original_date = date_match.group(1)

            normalized_date = normalize_date(original_date)

            time_match = TIME_REGEX.search(value)

            original_time = (
                time_match.group(0).strip()
                if time_match
                else None
            )

            if normalized_date:

                return (
                    normalized_date,
                    original_date,
                    original_time,
                )

    return None, None, None


# =========================================================
# HOSPITAL NAME EXTRACTION
# =========================================================


def extract_hospital_name(text: str) -> Optional[str]:

    """
    Extract complete hospital name from OCR header.

    Supports:
    CITY CARE
    LABORATORY MEDICINE
    MULTISPECIALITY HOSPITAL

    Result:
    CITY CARE MULTISPECIALITY HOSPITAL
    """

    lines = [
        clean(line)
        for line in text.splitlines()
    ]

    lines = [line for line in lines if line]

    # 1. Explicit hospital name label

    labelled = extract_labeled_value(
        text,
        [
            r"Hospital\s+Name",
            r"Name\s+of\s+Hospital",
            r"Laboratory\s+Name",
            r"Lab\s+Name",
            r"Clinic\s+Name",
        ],
    )

    if labelled:
        return labelled

    # 2. Identify hospital suffix

    suffix_pattern = re.compile(
        r"\b("
        r"multispeciality hospital|"
        r"multi speciality hospital|"
        r"super speciality hospital|"
        r"speciality hospital|"
        r"hospital|"
        r"medical centre|"
        r"medical center|"
        r"clinic|"
        r"diagnostic centre|"
        r"diagnostic center"
        r")\b",
        re.IGNORECASE,
    )

    ignored_lines = re.compile(
        r"^(DEPARTMENT OF|LABORATORY MEDICINE|"
        r"HAEMATOLOGY|HEMATOLOGY|BIOCHEMISTRY|"
        r"MICROBIOLOGY|PATHOLOGY|RADIOLOGY|"
        r"ACCURATE DIAGNOSIS|FOR A HEALTHIER TOMORROW)$",
        re.IGNORECASE,
    )

    noise_pattern = re.compile(
        r"(@|www\.|https?://|"
        r"\+?\d[\d\s()-]{7,}|"
        r"\b(address|phone|mobile|email|"
        r"road|street|pincode)\b)",
        re.IGNORECASE,
    )

    # 3. Search hospital suffix in header

    for index, line in enumerate(lines[:35]):

        if not suffix_pattern.search(line):
            continue

        if ignored_lines.fullmatch(line):
            continue

        if noise_pattern.search(line):
            continue

        # If current line contains only the suffix,
        # search backward for organization name.

        name_part = suffix_pattern.sub("", line).strip()

        if name_part:
            return clean(line)

        for previous_index in range(index - 1, max(-1, index - 8), -1):

            previous = lines[previous_index]

            if ignored_lines.fullmatch(previous):
                continue

            if noise_pattern.search(previous):
                continue

            if re.search(
                r"\b(compassion|better health|"
                r"accurate diagnosis|healthier tomorrow)\b",
                previous,
                re.IGNORECASE,
            ):
                continue

            if len(previous) < 3:
                continue

            if re.search(r"\d", previous):
                continue

            return clean(previous + " " + line)

        return clean(line)

    return None



# =========================================================
# DOCUMENT TYPE CLASSIFICATION
# =========================================================

def extract_document_type(
    text: str,
    filename: str = ""
) -> str:

    combined = f"{filename}\n{text}".lower()

    # IMPORTANT:
    # Lab report is checked before discharge summary.
    # A lab report may contain admission/discharge dates.

    rules = [

        ("lab_report", [
            "laboratory medicine",
            "laboratory report",
            "lab report",
            "pathology report",
            "haematology",
            "hematology",
            "complete blood count",
            "cbc & platelet",
            "reference range",
            "blood report",
            "blood_report",
            "blood test",
            "lab_result",
            "test parameter",
            "biological reference range",
        ]),

        ("prescription", [
            "prescription",
            "rx:",
            "medication prescribed",
            "dosage",
            "take one tablet",
        ]),

        ("discharge_summary", [
            "discharge summary",
            "discharge diagnosis",
            "discharged on",
        ]),

        ("admission_record", [
            "admission record",
            "admission date",
            "date of admission",
            "admitted on",
        ]),

        ("radiology_report", [
            "radiology report",
            "x-ray report",
            "xray report",
            "ultrasound report",
            "mri report",
            "ct scan report",
        ]),

        ("medical_bill", [
            "hospital bill",
            "medical bill",
            "invoice",
            "bill amount",
            "total payable",
            "receipt",
        ]),

        ("medical_certificate", [
            "medical certificate",
            "fitness certificate",
            "sick leave certificate",
        ]),
    ]

    for document_type, keywords in rules:

        if any(keyword in combined for keyword in keywords):
            return document_type

    if any(term in combined for term in [
        "medical report",
        "clinical report",
        "clinical findings",
        "medical history",
        "patient history",
        "diagnostic report",
        "doctor's notes",
    ]):
        return "medical_report"

    return "unknown"


# =========================================================
# MAIN METADATA EXTRACTION
# =========================================================

def extract_document_metadata(
    text: str,
    filename: str = ""
) -> dict:

    fields = {}

    # -----------------------------------------------------
    # PATIENT NAME
    # -----------------------------------------------------

    patient_name = extract_labeled_value(
        text,
        [
            r"Patient\s+Name",
            r"Name\s+of\s+Patient",
            r"Patient\s+Full\s+Name",
            r"Name\s+of\s+the\s+Patient",
        ],
    )

    if patient_name:
        fields["patient_name"] = patient_name

    # -----------------------------------------------------
    # HOSPITAL NAME
    # -----------------------------------------------------

    hospital_name = extract_hospital_name(text)

    if hospital_name:
        fields["hospital_name"] = hospital_name

    # -----------------------------------------------------
    # DOCTOR NAME
    # -----------------------------------------------------

    doctor_name = extract_labeled_value(
        text,
        [
            r"Primary\s+Consultant",
            r"Attending\s+Physician",
            r"Consultant\s+Doctor",
            r"Treating\s+Doctor",
            r"Doctor\s+Name",
            r"Physician\s+Name",
            r"Referring\s+Doctor",
            r"Referred\s+By",
        ],
    )

    if doctor_name:
        fields["doctor_name"] = doctor_name

    # -----------------------------------------------------
    # DATE DEFINITIONS
    # -----------------------------------------------------

    date_definitions = {

        "report_date": [
            r"Report\s+Date(?:\s*/\s*Time)?",
            r"Date\s+of\s+Report",
            r"Reported\s+On",
            r"Result\s+Date",
        ],

        "document_date": [
            r"Document\s+Date",
            r"Date\s+of\s+Document",
            r"Visit\s+Date",
            r"Consultation\s+Date",
            r"Examination\s+Date",
        ],

        "sample_collection_date": [
            r"Sample\s+Collected\s+On",
            r"Sample\s+Collection(?:\s+Date)?",
            r"Collection\s+Date",
            r"Collected\s+On",
            r"Specimen\s+Collection\s+Date",
        ],

        "admission_date": [
            r"Admission\s+Date",
            r"Admit\s+Date",
            r"Date\s+of\s+Admission",
            r"Admitted\s+On",
            r"DOA",
        ],

        "order_date": [
            r"Order\s+Date(?:\s*/\s*Time)?",
            r"Date\s+of\s+Order",
        ],

        "discharge_date": [
            r"Discharge\s+Date",
            r"Date\s+of\s+Discharge",
            r"Discharged\s+On",
            r"DOD",
        ],
    }

    # -----------------------------------------------------
    # EXTRACT ALL AVAILABLE DATES
    # -----------------------------------------------------

    for field, labels in date_definitions.items():

        normalized, original, time_value = extract_labeled_date(
            text,
            labels,
        )

        if normalized:

            fields[field] = normalized

            fields[f"{field}_original"] = original

        if time_value:

            time_field = field.replace("_date", "_time")

            fields[time_field] = time_value

    # -----------------------------------------------------
    # TIMELINE DATE PRIORITY
    # -----------------------------------------------------

    timeline_priority = [

        ("report_date", "high"),

        ("document_date", "high"),

        ("sample_collection_date", "medium"),

        ("admission_date", "medium"),

        ("order_date", "medium"),

        ("discharge_date", "medium"),
    ]

    timeline = {
        "date": None,
        "date_source": None,
        "confidence": None,
    }

    for field, confidence in timeline_priority:

        if fields.get(field):

            timeline = {
                "date": fields[field],
                "date_source": field,
                "confidence": confidence,
            }

            break

    # -----------------------------------------------------
    # DOCUMENT TYPE
    # -----------------------------------------------------

    document_type = extract_document_type(
        text,
        filename,
    )

    # -----------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------

    return {
        "document_type": document_type,
        "fields": fields,
        "timeline": timeline,
    }
