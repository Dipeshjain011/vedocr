import re
from typing import Any, Optional

MISSING_VALUES = {"", "-", "—", "–", "n/a", "na", "none", "null", "not available", "not provided", "not recorded"}

HEADER_ALIASES = {
    "test_name": {"parameter", "test", "test name", "test / panel", "investigation", "investigation name", "test / investigation"},
    "result": {"result", "results", "value", "reported result"},
    "reference_range": {"reference range", "biological ref range", "biological ref ranges", "biological reference range", "biological reference ranges", "normal range", "reference", "reference / range"},
    "unit": {"unit", "units"}, "reported_flag": {"flag", "reported flag"},
    "method": {"method", "methodology"}, "sample_type": {"specimen", "sample type"},
    "status": {"status", "test status"},
}

SECTION_HEADINGS = {
    "rbc parameters", "wbc parameters", "wbc differential and absolute counts", "platelet parameters",
    "rbc morphology", "wbc morphology", "platelet morphology", "lipid profile",
    "cbc (complete blood count), blood", "clinical information", "medications",
    "diagnostic report", "diagnostic reports", "comment", "summary", "end of report",
}
METHOD_ONLY = {
    "calculated", "measured", "spectrophotometric", "elect. impedance", "elect impedance",
    "impedance", "modified ifcc", "ifcc", "modified jaffe", "jaffe", "idms traceable",
    "vanadate oxidation", "enzymatic", "enzymatic colorimetric", "colorimetric", "chod-pod",
    "elimination/catalase", "catalase", "pod", "method", "methodology",
}
METHOD_TERMS = re.compile(r"\b(calculated|measured|spectrophotometric|impedance|colorimetric|enzymatic|ifcc|jaffe|oxidation|chod-pod|catalase|idms)\b", re.I)
FOOTER_TERMS = re.compile(r"(healthline|e-?mail\s*:|website\s*:|www\.|@|\bcin\s*(?:no\.?|number)\s*:|mumbai office|laboratory\s*:|end of report|authenticity assured|important instructions|page\s+\d+\s+of\s+\d+)", re.I)
UNIT_RE = re.compile(r"(?:mg/dl|mg/dL|g/dl|g/dL|U/L|IU/L|mIU/L|mmol/L|µmol/L|umol/L|fL|pg|%|/cmm|/cumm|mil/cemm|/emm|ml/min/1\.73sqm|ratio)\s*$", re.I)


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = normalize(value).strip(" \t:|")
    return None if value.casefold() in MISSING_VALUES else (value or None)


def detect_status(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    low = normalize(value).casefold()
    if any(x in low for x in ("not performed", "not done", "not tested", "not assessed", "not measured")):
        return "not_performed"
    if low in {"pending", "awaiting result", "in progress"}:
        return "pending"
    return None


def normalized_header(value: str) -> Optional[str]:
    low = normalize(value).casefold().strip(" :")
    for field, aliases in HEADER_ALIASES.items():
        if low in aliases:
            return field
    return None


def is_section_heading(value: str) -> bool:
    low = normalize(value).casefold().strip(" :")
    return low in SECTION_HEADINGS or bool(re.fullmatch(r"page\s+\d+(?:\s+of\s+\d+)?", low))


def is_reference_range(value: str) -> bool:
    value = normalize(value)
    has_number = bool(re.search(r"\d", value))
    numeric_range = bool(re.search(r"(?:<|>|<=|>=|≤|≥)?\s*\d+(?:\.\d+)?\s*(?:-|–|—|\bto\b)\s*(?:<|>|<=|>=|≤|≥)?\s*\d+(?:\.\d+)?", value, re.I))
    threshold = bool(re.search(r"^(?:<|>|<=|>=|≤|≥|</=|>/=)\s*\d", value))
    labelled = bool(re.search(r"\b(?:desirable|borderline|optimal|normal|high|low|very high|mild decrease|moderate|severe|kidney failure)\s*:", value, re.I))
    return has_number and (numeric_range or threshold or labelled)


def is_numeric_result(value: str) -> bool:
    value = normalize(value).replace(",", "")
    return bool(re.fullmatch(r"[<>≤≥]?\s*-?\d+(?:\.\d+)?\s*%?", value))


def is_method(value: str) -> bool:
    low = normalize(value).casefold().strip(" .,:;")
    return low in METHOD_ONLY or bool(METHOD_TERMS.search(low)) and len(low) < 55


def is_probable_test_name(value: str) -> bool:
    value = clean_value(value)
    if not value or len(value) > 100:
        return False
    low = value.casefold().strip(" :")
    if normalized_header(value) or is_section_heading(value) or FOOTER_TERMS.search(value):
        return False
    if is_numeric_result(value) or is_reference_range(value) or detect_status(value) or is_method(value):
        return False
    # OCR often turns units, method fragments, and eGFR interpretation text into labels.
    if re.fullmatch(r"(?:mg/?dl|g/?dl|u/l|iu/l|%|ratio|elimination/?|calculated|measured)", low):
        return False
    if re.search(r"\b(?:mild|moderate|severe)\s+(?:to\s+)?(?:moderate|severe|decrease|decrease in|reduction|kidney|renal)\b", low):
        return False
    if low.startswith(("elimination/", "mg/dl", "mg/dl ")):
        return False
    if not re.search(r"[A-Za-z]", value):
        return False
    if re.search(r"\b(?:email|website|healthline|phone|telephone|address|cin no|uhid|lab no|report status|collected at|processed at|page \d+)\b", value, re.I):
        return False
    # Reject likely metadata and long explanatory prose, while allowing test labels with commas/parentheses.
    if len(value.split()) > 10 or value.endswith((".", ";")):
        return False
    return True


def _extract_unit(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = re.search(r"(?:mg/dL|mg/dl|g/dL|g/dl|U/L|IU/L|mIU/L|mmol/L|µmol/L|umol/L|fL|pg|%|/cmm|/cumm|mil/cemm|/emm|ml/min/1\.73sqm|Ratio)\s*$", value, re.I)
    return match.group(0) if match else None


def _build_result(test_name: str, result: Optional[str], reference_range: Optional[str] = None,
                  method: Optional[str] = None, sample_type: Optional[str] = None,
                  reported_flag: Optional[str] = None, source_text: Optional[str] = None,
                  notes: Optional[str] = None) -> Optional[dict[str, Any]]:
    test_name = clean_value(test_name)
    if not test_name or not is_probable_test_name(test_name):
        return None
    result = clean_value(result)
    reference_range = clean_value(reference_range)
    method = clean_value(method)
    status = detect_status(result)
    if status == "not_performed":
        result = None
    # Do not mistake a reference interval for a measured result.
    if result and is_reference_range(result):
        result = None
    if not any([result, status, reference_range, method, reported_flag, notes]):
        return None
    return {
        "test_name": test_name, "category": None, "result": result,
        "unit": _extract_unit(reference_range), "reference_range": reference_range,
        "reported_flag": clean_value(reported_flag), "status": status or ("reported" if result else None),
        "sample_type": clean_value(sample_type), "collection_date": None, "method": method,
        "notes": clean_value(notes), "source_text": clean_value(source_text),
    }


def _split_row(line: str) -> list[str]:
    if "|" in line or "\t" in line:
        return [normalize(c) for c in re.split(r"\||\t+", line) if normalize(c)]
    if re.search(r"\s{2,}", line):
        return [normalize(c) for c in re.split(r"\s{2,}", line.strip()) if normalize(c)]
    return []


def extract_lab_results(text: str) -> list[dict[str, Any]]:
    """Heuristic lab-table extraction. Review output against the source report."""
    if not text or not text.strip():
        return []
    lines = [normalize(x) for x in text.replace("\r", "\n").splitlines() if normalize(x)]
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(item: Optional[dict[str, Any]]) -> None:
        if not item:
            return
        key = (item["test_name"].casefold(), (item.get("result") or "").casefold())
        if key not in seen:
            results.append(item)
            seen.add(key)

    # First handle rows where OCR preserved table columns on one line.
    for line in lines:
        if FOOTER_TERMS.search(line):
            continue
        cells = _split_row(line)
        if len(cells) < 2 or normalized_header(cells[0]):
            continue
        name = cells[0]
        if not is_probable_test_name(name):
            continue
        result = reference = method = flag = None
        for cell in cells[1:]:
            if normalized_header(cell):
                continue
            if is_reference_range(cell):
                reference = f"{reference}; {cell}" if reference and cell not in reference else (reference or cell)
            elif is_numeric_result(cell) and result is None:
                result = cell
            elif cell.casefold() in {"high", "low", "normal", "abnormal", "h", "l", "n"}:
                flag = cell
            elif is_method(cell) and method is None:
                method = cell
        add(_build_result(name, result, reference, method, reported_flag=flag, source_text=line))

    # Then handle OCR's common one-cell-per-line layout inside test tables only.
    in_table = False
    table_lines: list[str] = []
    for line in lines:
        low = line.casefold().strip(" :")
        if FOOTER_TERMS.search(line):
            in_table = False
            continue
        if low in {"parameter", "test", "investigation"}:
            in_table = True
            continue
        if normalized_header(line) and in_table:
            continue
        if is_section_heading(line):
            if low in {"lipid profile", "rbc parameters", "wbc parameters", "platelet parameters", "wbc differential and absolute counts"}:
                in_table = True
            elif low not in {"comment"}:
                in_table = False
            continue
        if in_table:
            table_lines.append(line)

    # Identify test labels, then only inspect cells until the next test label.
    for i, line in enumerate(table_lines):
        if not is_probable_test_name(line):
            continue
        # Don't re-add an inline row's test/result pair.
        result = reference = method = flag = sample_type = None
        captured = [line]
        for candidate in table_lines[i + 1:i + 6]:
            if is_section_heading(candidate) or normalized_header(candidate):
                break
            if is_probable_test_name(candidate):
                break
            captured.append(candidate)
            if is_numeric_result(candidate) and result is None:
                result = candidate
            elif is_reference_range(candidate):
                reference = f"{reference}; {candidate}" if reference and candidate not in reference else (reference or candidate)
            elif candidate.casefold() in {"high", "low", "normal", "abnormal", "h", "l", "n"}:
                flag = candidate
            elif re.search(r"\b(?:serum|plasma|whole blood|urine|edta)\b", candidate, re.I):
                sample_type = candidate
            elif is_method(candidate) and method is None:
                method = candidate
        # Never borrow the preceding numeric value here: in crowded OCR tables it
        # commonly belongs to the previous test (e.g., absolute cell counts).
        add(_build_result(line, result, reference, method, sample_type, flag, " | ".join(captured)))

    return results
