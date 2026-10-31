import re
from typing import Optional


# --------------------------------------------------
# GENERAL CLEANING
# --------------------------------------------------

def clean_value(value: Optional[str]) -> Optional[str]:
    """Normalize whitespace and remove surrounding punctuation."""
    if not value:
        return None

    value = re.sub(r"\s+", " ", value).strip(" \t\r\n:-")
    return value or None


def is_section_boundary(line: str) -> bool:
    """Detect numbered sections and common standalone headings."""
    line = line.strip()

    if not line:
        return False

    if re.match(r"^\d{1,2}\s*[.):\-]\s+", line):
        return True

    heading_pattern = (
        r"^(?:"
        r"diagnostic report|prescription(?:\s*/\s*medication)?|"
        r"medication section|clinical information|"
        r"laboratory report|lab report|lab results|laboratory results|medications?|medicines?|vital signs|"
        r"medical history|past medical history|"
        r"disclaimer|signature|end of report|"
        r"clinical notes|vital signs\s*&\s*body measurements"
        r")(?:\s*\(.*\))?\s*:?\s*$"
    )

    return bool(re.match(heading_pattern, line, re.IGNORECASE))


def remove_inline_section_heading(value: str) -> str:
    """Remove a numbered section heading appended to a value."""
    if not value:
        return ""

    pattern = (
        r"\s+\d{1,2}\s*[.):\-]\s+"
        r"(?=(?:diagnostic|prescription|medication|"
        r"clinical information|laboratory report|lab report|"
        r"disclaimer|vital signs|clinical notes)\b)"
    )

    return re.split(pattern, value, maxsplit=1, flags=re.IGNORECASE)[0].strip()


# --------------------------------------------------
# LABELED FIELD EXTRACTION
# --------------------------------------------------

def find_labeled_value(
    text: str,
    labels: list[str],
) -> Optional[str]:
    """Extract a value written beside a label."""
    for label in labels:
        pattern = rf"(?im)^\s*(?:{label})\s*[:\-]\s*(.+?)\s*$"
        match = re.search(pattern, text)

        if match:
            value = clean_value(match.group(1))
            if value:
                value = remove_inline_section_heading(value)
                value = clean_value(value)

            if value:
                return value

    return None


# --------------------------------------------------
# SECTION EXTRACTION
# --------------------------------------------------

def extract_section(
    text: str,
    headings: list[str],
    stop_headings: list[str],
) -> Optional[str]:
    """Extract a section and stop at the next recognized heading."""        
    lines = text.splitlines()

    heading_pattern = re.compile(
        r"^\s*(?:" + "|".join(headings) + r")\s*:?\s*$",
        re.IGNORECASE,
    )

    stop_pattern = re.compile(
        r"^\s*(?:" + "|".join(stop_headings) + r")\s*:?\s*$",
        re.IGNORECASE,
    )

    start = None
    for index, line in enumerate(lines):
        if heading_pattern.match(line.strip()):
            start = index + 1
            break

    if start is None:
        return None

    collected = []
    for line in lines[start:]:
        stripped = line.strip()

        if not stripped:
            continue

        if is_section_boundary(stripped):
            break

        if stop_pattern.match(stripped):
            break

        cleaned_line = remove_inline_section_heading(stripped)
        if cleaned_line:
            collected.append(cleaned_line)

        if cleaned_line != stripped:
            break

    return clean_value(" ".join(collected))


# --------------------------------------------------
# LAB RESULTS
# --------------------------------------------------

def extract_lab_results(text: str) -> list[dict]:
    """
    Extract line-oriented lab tables containing:
    test/parameter, result, unit, reference, flag.

    Supports numeric and text results (e.g. Negative, Clear, Pale yellow)   
    and flags split across lines (e.g. Near + optimal).
    """
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    results: list[dict] = []
    seen: set[tuple[str, str]] = set()

    test_headers = {"test name", "parameter", "field"}
    unit_headers = {"unit", "unit / details"}
    reference_headers = {
        "reference range",
        "reference / interpretation",
        "reference / expected",
        "reference range / interpretation",
        "reference / expected range",
    }

    section_titles = re.compile(
        r"^(?:"
        r"\d{1,2}\s*[.)\-]\s+|"
        r"vital signs(?:\s*&\s*body measurements)?|"
        r"clinical notes|diagnostic report|prescription|"
        r"important\s*:|"
        r"comprehensive sample laboratory report|"
        r"patient\s*&\s*report information"
        r")",
        re.IGNORECASE,
    )

    def is_table_header(index: int) -> bool:
        if index + 4 >= len(lines):
            return False

        header = [x.casefold() for x in lines[index:index + 5]]
        return (
            header[0] in test_headers
            and header[1] == "result"
            and header[2] in unit_headers
            and header[3] in reference_headers
            and header[4] == "flag"
        )

    def looks_like_next_test(index: int) -> bool:
        """Heuristic to avoid reading a section heading as a table row."""  
        if index >= len(lines):
            return True
        candidate = lines[index]
        return bool(
            section_titles.match(candidate)
            or candidate.casefold() in test_headers | {
                "result", "unit", "unit / details", "flag",
            }
        )

    i = 0
    while i < len(lines):
        if not is_table_header(i):
            i += 1
            continue

        i += 5

        while i < len(lines):
            # If another table header follows, start parsing it too.        
            if is_table_header(i):
                i += 5
                continue

            name = lines[i].strip()

            if looks_like_next_test(i):
                break

            # A table row has five consecutive cells. Text values are valid.
            if i + 4 >= len(lines):
                break

            value = lines[i + 1].strip()
            unit = lines[i + 2].strip()
            reference = lines[i + 3].strip()
            flag = lines[i + 4].strip()

            if any(looks_like_next_test(j) for j in (i + 1, i + 2, i + 3, i + 4)):
                break

            # PDF text extraction sometimes splits a flag across lines:     
            # "Near", "optimal", then the next test name.
            if (
                flag.casefold() == "near"
                and i + 5 < len(lines)
                and lines[i + 5].casefold() == "optimal"
            ):
                flag = "Near optimal"
                i += 1

            if not name or not value:
                i += 5
                continue

            key = (name.casefold(), value.casefold())
            if key not in seen:
                results.append({
                    "test_name": name,
                    "result": value,
                    "unit": None if unit in {"-", "—", "–", "â€”"} else unit,
                    "reference_range": reference,
                    "reported_flag": None if flag in {"-", "—", "–", "â€”"} else flag,
                    "sample_type": None,
                    "collection_date": None,
                })
                seen.add(key)

            i += 5

    return results


# --------------------------------------------------
# VITAL SIGNS
# --------------------------------------------------

def extract_vitals(text: str) -> dict:
    """Extract explicitly labelled vital signs."""
    normalized = re.sub(r"[ \t]+", " ", text)

    patterns = {
        "blood_pressure": (
            r"\b(?:blood pressure|BP)\s*[:\-]?\s*"
            r"(\d{2,3}\s*/\s*\d{2,3}\s*(?:mmHg)?)"
        ),
        "pulse_rate": (
            r"\b(?:pulse(?: rate)?|heart rate)\s*[:\-]?\s*"
            r"(\d{2,3}\s*(?:bpm|beats/min)?)"
        ),
        "temperature": (
            r"\btemperature\s*[:\-]?\s*"
            r"(\d{2,3}(?:\.\d+)?\s*(?:Â°?\s*[CF])?)"
        ),
        "spo2": (
            r"\b(?:SpO2|oxygen saturation)\s*[:\-]?\s*"
            r"(\d{2,3}\s*%?)"
        ),
        "respiratory_rate": (
            r"\brespiratory rate\s*[:\-]?\s*"
            r"(\d{1,3}\s*(?:breaths?\s*/\s*min|"
            r"breaths?\s+per\s+minute|/min)?)"
        ),
        "height": (
            r"\bheight\s*[:\-]?\s*"
            r"(\d{2,3}(?:\.\d+)?\s*(?:cm|m|inches|in)\b)"
        ),
        "weight": (
            r"\bweight\s*[:\-]?\s*"
            r"(\d{1,3}(?:\.\d+)?\s*(?:kg|g|lbs?)\b)"
        ),
        "bmi": (
            r"\bBMI\s*[:\-]?\s*"
            r"(\d{1,2}(?:\.\d+)?(?:\s*kg/mÂ²)?)"
        ),
        "blood_glucose": (
            r"\b(?:fasting blood glucose|blood glucose|"
            r"blood sugar|glucose)\s*[:\-]?\s*"
            r"(\d{2,3}(?:\.\d+)?\s*(?:mg/dL|mmol/L)?)"
        ),
    }

    output = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            value = clean_value(match.group(1))
            if value:
                output[field] = value

    # Handle labels and values split onto separate lines.
    lines = [line.strip() for line in text.splitlines() if line.strip()]    
    for index, line in enumerate(lines[:-1]):
        if line.casefold().rstrip(":") == "respiratory rate":
            if "respiratory_rate" not in output:
                candidate = lines[index + 1]
                match = re.match(
                    r"^(\d{1,3}\s*(?:breaths?\s*/\s*min|"
                    r"breaths?\s+per\s+minute|/min))$",
                    candidate,
                    re.IGNORECASE,
                )
                if match:
                    output["respiratory_rate"] = clean_value(match.group(1))

    return output


# --------------------------------------------------
# COMPLETE MEDICAL DATA EXTRACTION
# --------------------------------------------------

def extract_medical_data(text: str) -> dict:
    """Extract structured medical information from document text."""        

    clinical_stop = [
        r"diagnosis",
        r"past medical history",
        r"medical history",
        r"history of present illness",
        r"surgical history",
        r"allergies",
        r"treatment(?:\s*/\s*procedure)?",
        r"procedure",
        r"medications",
        r"prescription",
        r"vitals",
        r"investigations",
        r"findings",
        r"impression",
        r"conclusion",
        r"recommendations?",
        r"follow.?up",
        r"discharge instructions",
        r"^\d{1,2}\s*[.):\-]\s+.*",
    ]

    symptoms = extract_section(
        text,
        [
            r"chief complaints?",
            r"symptoms",
            r"presenting complaints?",
        ],
        clinical_stop,
    )

    history = extract_section(
        text,
        [
            r"medical history",
            r"past medical history",
            r"history of present illness",
        ],
        clinical_stop,
    )

    diagnosis = find_labeled_value(
        text,
        [
            r"final diagnosis",
            r"provisional diagnosis",
            r"diagnosis",
        ],
    )

    allergies = find_labeled_value(
        text,
        [
            r"allergies",
            r"drug allergies",
            r"known allergies",
        ],
    )

    treatment = extract_section(
        text,
        [
            r"treatment",
            r"treatment given",
            r"procedures performed",
            r"treatment\s*/\s*procedure",
        ],
        clinical_stop,
    )

    # Extract medicines from explicit labels and from a MEDICATIONS /
    # PRESCRIPTION section. Supports lines such as:
    # "Cetirizine - 10 mg - once daily (fictional test entry)"
    medications = []
    seen_medicines: set[str] = set()

    medication_heading = re.compile(
        r"^\s*(?:medications?|prescriptions?|medicines?)\s*:?\s*$",
        re.IGNORECASE,
    )
    medication_stop = re.compile(
        r"^\s*(?:vitals?|vital signs|lab results|laboratory results|"
        r"clinical notes|clinical information|diagnostic report|"
        r"diagnostic reports|conclusion|recommendations?|"
        r"sample OCR text|end of report|end of synthetic document)\b",
        re.IGNORECASE,
    )
    medicine_label = re.compile(
        r"^\s*(?:medicine|medication|drug)(?:\s+name)?\s*[:\-]\s*(.+?)\s*$",
        re.IGNORECASE,
    )

    def add_medicine(raw: str) -> None:
        raw = clean_value(raw)
        if not raw:
            return
        raw = remove_inline_section_heading(raw)
        raw = re.sub(r"\s*\(fictional test entry\)\s*$", "", raw, flags=re.IGNORECASE)
        raw = clean_value(raw)
        if not raw:
            return

        # Ignore headings, placeholders, and obvious non-medicine boilerplate.
        if raw.casefold() in {
            "medicine", "medication", "medications", "prescription",
            "drug", "name", "not available", "n/a", "none", "-",
        }:
            return

        # Split common separators while preserving the full instruction string.
        parts = [clean_value(part) for part in re.split(r"\s+[|]\s+|\s+[–—-]\s+", raw)]
        parts = [part for part in parts if part]
        medicine_name = parts[0] if parts else raw

        # For medicine entries like "Cetirizine - 10 mg - once daily",
        # the first segment is the name and the remaining segments are instructions.
        if not medicine_name or medicine_name.casefold() in seen_medicines:
            return

        strength_match = re.search(
            r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|mL|units?)\b",
            raw,
            re.IGNORECASE,
        )
        frequency_match = re.search(
            r"\b(once|twice|thrice|one|two|three)\s+(?:a|per)\s+day\b|"
            r"\b(?:once|twice|thrice)\s+daily\b|"
            r"\b\d+\s+times?\s+(?:a|per)\s+day\b",
            raw,
            re.IGNORECASE,
        )
        dosage_match = re.search(
            r"\b\d+\s*(?:tablet|tablets|capsule|capsules|puff|puffs|drop|drops)\b",
            raw,
            re.IGNORECASE,
        )
        duration_match = re.search(
            r"\bfor\s+\d+\s+(?:days?|weeks?|months?)\b",
            raw,
            re.IGNORECASE,
        )
        medications.append({
            "medicine_name_and_instructions": raw,
            "medicine_name": medicine_name,
            "strength": strength_match.group(0) if strength_match else None,
            "dosage": dosage_match.group(0) if dosage_match else None,
            "frequency": frequency_match.group(0) if frequency_match else None,
            "duration": duration_match.group(0) if duration_match else None,
            "route": None,
        })
        seen_medicines.add(medicine_name.casefold())

    in_medication_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if medication_heading.match(stripped):
            in_medication_section = True
            continue

        if in_medication_section and medication_stop.match(stripped):
            in_medication_section = False
            continue

        labelled_match = medicine_label.match(stripped)
        if labelled_match:
            add_medicine(labelled_match.group(1))
            continue

        if in_medication_section:
            # Do not treat generic headings or OCR noise as medicine entries.
            if (
                len(stripped) <= 100
                and re.search(r"[A-Za-z]", stripped)
                and not re.match(r"^(?:sample OCR text|this is a fictional|end of)", stripped, re.I)
            ):
                add_medicine(stripped)

    diagnostic_sections = {
        "findings": [
            r"findings",
            r"radiological findings",
        ],
        "impression": [
            r"impression",
            r"radiology impression",
        ],
        "conclusion": [
            r"conclusion",
            r"interpretation",
        ],
        "recommendations": [
            r"recommendations?",
            r"advice",
            r"follow.?up",
        ],
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
        r"medicine name",
        r"important:?",
        r"^\d{1,2}\s*[.):\-]\s+.*",
    ]

    diagnostic = {}
    for field, headings in diagnostic_sections.items():
        value = extract_section(text, headings, diagnostic_stops)
        if value:
            value = remove_inline_section_heading(value)
            value = clean_value(value)
        if value:
            diagnostic[field] = value

    clinical_information = {}
    for key, value in {
        "symptoms": symptoms,
        "diagnosis": diagnosis,
        "medical_history": history,
        "allergies": allergies,
        "treatment_and_procedures": treatment,
    }.items():
        if value:
            value = remove_inline_section_heading(value)
            value = clean_value(value)
        if value:
            clinical_information[key] = value

    return {
        "lab_results": extract_lab_results(text),
        "clinical_information": clinical_information,
        "medications": medications,
        "vitals": extract_vitals(text),
        "diagnostic_reports": diagnostic,
    }
