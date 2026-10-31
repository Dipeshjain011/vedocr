
import re
from typing import Any, Optional

from services.lab_results_extractor import extract_lab_results


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    value = re.sub(r"\s+", " ", value).strip(" \t:|")

    if not value or value.casefold() in {
        "n/a", "na", "none", "null", "not provided", "-"
    }:
        return None

    return value


def _extract_label(
    text: str,
    labels: list[str],
) -> Optional[str]:
    """Extract a value written beside a label."""
    for label in labels:
        pattern = (
            rf"(?im)^\s*(?:{label})\s*[:：]\s*(.*?)\s*$"
        )
        match = re.search(pattern, text)

        if match:
            value = _clean(match.group(1))
            if value:
                return value

    return None


def _extract_section(
    text: str,
    headings: list[str],
    stop_headings: list[str],
) -> Optional[str]:
    """Collect section text until another recognized heading."""
    lines = text.splitlines()

    heading_re = re.compile(
        r"^\s*(?:" + "|".join(headings) + r")\s*:?\s*$",
        re.IGNORECASE,
    )
    stop_re = re.compile(
        r"^\s*(?:" + "|".join(stop_headings) + r")\s*:?\s*$",
        re.IGNORECASE,
    )

    start = None

    for index, line in enumerate(lines):
        if heading_re.fullmatch(line.strip()):
            start = index + 1
            break

    if start is None:
        return None

    collected = []

    for line in lines[start:]:
        line = line.strip()

        if not line:
            continue

        if stop_re.fullmatch(line):
            break

        if re.match(r"^\d{1,2}[.)]\s+", line):
            break

        collected.append(line)

    return _clean(" ".join(collected))


def extract_vitals(text: str) -> dict[str, Any]:
    """Extract explicitly labelled vital signs and measurements."""
    normalized = re.sub(r"[ \t]+", " ", text)

    patterns = {
        "blood_pressure": (
            r"\b(?:blood pressure|BP)\s*[:：-]?\s*"
            r"(\d{2,3}\s*/\s*\d{2,3}\s*(?:mmHg)?)"
        ),
        "pulse_rate": (
            r"\b(?:pulse(?: rate)?|heart rate)\s*[:：-]?\s*"
            r"(\d{2,3}\s*(?:bpm|beats/min)?)"
        ),
        "temperature": (
            r"\btemperature\s*[:：-]?\s*"
            r"(\d{2,3}(?:\.\d+)?\s*(?:°?\s*[CF])?)"
        ),
        "spo2": (
            r"\b(?:SpO2|oxygen saturation)\s*[:：-]?\s*"
            r"(\d{2,3}\s*%?)"
        ),
        "respiratory_rate": (
            r"\brespiratory rate\s*[:：-]?\s*"
            r"(\d{1,3}\s*(?:breaths?\s*/\s*min|"
            r"breaths?\s+per\s+minute|/min)?)"
        ),
        "height": (
            r"\bheight\s*[:：-]?\s*"
            r"(\d{2,3}(?:\.\d+)?\s*(?:cm|m|inches|in)\b)"
        ),
        "weight": (
            r"\bweight\s*[:：-]?\s*"
            r"(\d{1,3}(?:\.\d+)?\s*(?:kg|g|lbs?)\b)"
        ),
        "bmi": (
            r"\bBMI\s*[:：-]\s*"
            r"(\d{1,2}(?:\.\d+)?(?:\s*kg/m²)?)"
        ),
        "blood_glucose": (
            r"\b(?:fasting blood glucose|blood glucose|"
            r"blood sugar|glucose)\s*[:：-]\s*"
            r"(\d{2,3}(?:\.\d+)?\s*(?:mg/dL|mmol/L)?)"
        ),
    }

    output = {}

    for field, pattern in patterns.items():
        match = re.search(pattern, normalized, re.IGNORECASE)

        if match:
            value = _clean(match.group(1))
            if value:
                output[field] = value

    # Handle common cell-per-line layouts:
    # Blood pressure / 118/76 mmHg
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    label_map = {
        "blood pressure": "blood_pressure",
        "pulse / heart rate": "pulse_rate",
        "pulse": "pulse_rate",
        "body temperature": "temperature",
        "temperature": "temperature",
        "respiratory rate": "respiratory_rate",
        "oxygen saturation (spo2)": "spo2",
        "oxygen saturation": "spo2",
        "height": "height",
        "weight": "weight",
        "bmi": "bmi",
    }

    for index, line in enumerate(lines[:-1]):
        key = label_map.get(line.casefold().rstrip(":"))

        if key and key not in output:
            candidate = _clean(lines[index + 1])

            if candidate and not re.match(
                r"^(?:not measured|not assessed|not recorded|"
                r"not performed|not interpreted)\b",
                candidate,
                re.IGNORECASE,
            ):
                # Only accept a value containing a digit.
                if re.search(r"\d", candidate):
                    output[key] = candidate

    # Never calculate BMI from height and weight here.
    # Preserve only a BMI explicitly stated in the document.
    return output


def extract_medical_data(text: str) -> dict[str, Any]:
    """Extract structured medical data without inventing results."""

    # 1. Lab tests
    lab_results = extract_lab_results(text)

    # 2. Vital signs
    vitals = extract_vitals(text)

    # 3. Clinical information
    clinical_information = {}

    clinical_labels = {
        "diagnosis": [
            r"final diagnosis",
            r"provisional diagnosis",
            r"diagnosis",
        ],
        "allergies": [
            r"known allergies",
            r"drug allergies",
            r"allergies",
        ],
        "reason_for_visit": [
            r"reason for visit",
            r"chief complaint",
            r"reason for consultation",
        ],
        "medical_condition": [
            r"medical condition",
            r"presenting condition",
        ],
    }

    for field, labels in clinical_labels.items():
        value = _extract_label(text, labels)
        if value:
            clinical_information[field] = value

    clinical_sections = {
        "symptoms": [
            r"chief complaints?",
            r"presenting complaints?",
            r"symptoms",
        ],
        "medical_history": [
            r"past medical history",
            r"medical history",
            r"history of present illness",
        ],
        "treatment_and_procedures": [
            r"treatment given",
            r"treatment",
            r"procedures performed",
        ],
    }

    section_stops = [
        r"diagnosis",
        r"medical history",
        r"medications?",
        r"prescription",
        r"vitals?",
        r"vital signs",
        r"investigations",
        r"findings",
        r"impression",
        r"conclusion",
        r"recommendations?",
        r"follow.?up",
        r"discharge instructions",
        r"disclaimer",
        r"summary",
    ]

    for field, headings in clinical_sections.items():
        value = _extract_section(text, headings, section_stops)
        if value:
            clinical_information.setdefault(field, value)

    # 4. Medication extraction
    medications = []
    in_medication_section = False
    medication_stop = re.compile(
        r"^\s*(?:vitals?|vital signs|lab results|laboratory results|"
        r"clinical notes|clinical information|diagnostic reports?|"
        r"conclusion|recommendations?|summary|disclaimer)\s*:?\s*$",
        re.IGNORECASE,
    )

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if re.fullmatch(
            r"(?:medications?|prescriptions?|medicines?)\s*:?",
            line,
            re.IGNORECASE,
        ):
            in_medication_section = True
            continue

        if in_medication_section and medication_stop.fullmatch(line):
            in_medication_section = False
            continue

        match = re.match(
            r"^(?:medicine|medication|drug)(?:\s+name)?\s*[:：-]\s*(.+)$",
            line,
            re.IGNORECASE,
        )

        if match:
            raw = _clean(match.group(1))
        elif in_medication_section:
            raw = _clean(line)
        else:
            continue

        if not raw or raw.casefold() in {
            "medicine", "medication", "medications",
            "medicine name", "strength", "frequency",
            "dose", "duration", "instructions",
            "not available", "none",
        }:
            continue

        if len(raw) > 180:
            continue

        # Avoid adding section titles or explanatory paragraphs.
        if re.match(
            r"^(?:sample OCR text|this is a fictional|"
            r"important note|end of report|disclaimer)\b",
            raw,
            re.IGNORECASE,
        ):
            continue

        if not any(
            item["medicine_name_and_instructions"].casefold()
            == raw.casefold()
            for item in medications
        ):
            strength = re.search(
                r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|mL|units?)\b",
                raw,
                re.IGNORECASE,
            )
            frequency = re.search(
                r"\b(?:once|twice|thrice)\s+(?:daily|a day)|"
                r"\b\d+\s+times?\s+(?:daily|a day|per day)|"
                r"\bat night\b|\bas needed\b",
                raw,
                re.IGNORECASE,
            )
            duration = re.search(
                r"\bfor\s+\d+\s+(?:days?|weeks?|months?)\b",
                raw,
                re.IGNORECASE,
            )

            medications.append({
                "medicine_name_and_instructions": raw,
                "strength": strength.group(0) if strength else None,
                "dosage": None,
                "frequency": frequency.group(0) if frequency else None,
                "duration": duration.group(0) if duration else None,
                "route": None,
            })

    # 5. Diagnostic findings
    diagnostic_reports = {}

    diagnostic_sections = {
        "findings": [r"findings", r"radiological findings"],
        "impression": [r"impression", r"radiology impression"],
        "conclusion": [r"conclusion", r"interpretation"],
        "recommendations": [r"recommendations?", r"advice", r"follow.?up"],
    }

    diagnostic_stops = [
        r"findings",
        r"impression",
        r"conclusion",
        r"recommendations?",
        r"advice",
        r"follow.?up",
        r"signature",
        r"doctor name",
        r"prescription",
        r"medications?",
        r"vital signs",
        r"summary",
        r"disclaimer",
    ]

    for field, headings in diagnostic_sections.items():
        value = _extract_section(text, headings, diagnostic_stops)
        if value:
            diagnostic_reports[field] = value

    return {
        "lab_results": lab_results,
        "clinical_information": clinical_information,
        "medications": medications,
        "vitals": vitals,
        "diagnostic_reports": diagnostic_reports,
    }
