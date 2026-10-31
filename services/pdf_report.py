
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#17365D")
TEAL = colors.HexColor("#117C83")
LIGHT_GREY = colors.HexColor("#F3F5F7")
BORDER = colors.HexColor("#D5DCE5")
DARK_GREY = colors.HexColor("#454B54")
WARNING_BG = colors.HexColor("#FFF4D6")
WARNING_TEXT = colors.HexColor("#8A4B08")


def _display(value: Any) -> str:
    """Convert extracted values to safe text for ReportLab."""

    if value is None or value == "":
        return "Not available"

    if isinstance(value, dict):
        return "; ".join(
            f"{key.replace('_', ' ').title()}: {_display(item)}"
            for key, item in value.items()
        ) or "Not available"

    if isinstance(value, (list, tuple)):
        return "; ".join(_display(item) for item in value) or "None"

    text = str(value).replace("\x00", "")
    return escape(text).replace("\n", "<br/>")


def _p(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_display(value), style)


def _section(title: str, styles: dict) -> Paragraph:
    return Paragraph(escape(title), styles["section"])


def _table(
    rows: list[list[Any]],
    widths: list[float],
    styles: dict,
    header: bool = True,
) -> Table:
    """Create a formatted table with repeating header rows."""

    converted = []

    for row_index, row in enumerate(rows):
        converted_row = []

        for value in row:
            if isinstance(value, Paragraph):
                converted_row.append(value)
            else:
                style_name = (
                    "table_header"
                    if header and row_index == 0
                    else "table_cell"
                )
                converted_row.append(_p(value, styles[style_name]))

        converted.append(converted_row)

    table = Table(
        converted,
        colWidths=widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
        splitByRow=1,
    )

    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]

    if header and converted:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])

        if len(converted) > 1:
            commands.append(
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, LIGHT_GREY],
                )
            )
    elif converted:
        commands.append(
            (
                "ROWBACKGROUNDS",
                (0, 0),
                (-1, -1),
                [colors.white, LIGHT_GREY],
            )
        )

    table.setStyle(TableStyle(commands))
    return table


def _footer(canvas, doc):
    """Draw page numbers and a verification notice."""

    canvas.saveState()

    page_width, _ = A4
    canvas.setStrokeColor(BORDER)
    canvas.line(18 * mm, 15 * mm, page_width - 18 * mm, 15 * mm)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(DARK_GREY)
    canvas.drawString(
        18 * mm,
        10 * mm,
        "VedAmrit | OCR output - manual verification required",
    )
    canvas.drawRightString(
        page_width - 18 * mm,
        10 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


def create_medical_report_pdf(data: dict[str, Any]) -> bytes:
    """
    Generate a professional PDF from the existing OCR response.
    Returns PDF bytes; it does not modify the OCR JSON response.
    """

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="VedAmrit Medical OCR Report",
        author="VedAmrit",
    )

    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "VedAmritTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "VedAmritSubtitle",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
            textColor=DARK_GREY,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "section": ParagraphStyle(
            "VedAmritSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "VedAmritBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=DARK_GREY,
        ),
        "table_header": ParagraphStyle(
            "VedAmritTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            "VedAmritTableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=DARK_GREY,
            wordWrap="CJK",
        ),
        "notice": ParagraphStyle(
            "VedAmritNotice",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=WARNING_TEXT,
            backColor=WARNING_BG,
            borderColor=colors.HexColor("#E7C66A"),
            borderWidth=0.5,
            borderPadding=7,
            spaceBefore=7,
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "VedAmritSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=DARK_GREY,
        ),
    }

    story = []

    # Report heading
    story.append(Paragraph("VedAmrit", styles["title"]))
    story.append(
        Paragraph(
            "Medical Document Extraction Report",
            styles["subtitle"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Source file:</b> "
            f"{_display(data.get('filename', 'Uploaded document'))}",
            styles["body"],
        )
    )
    story.append(Spacer(1, 5))

    # Verification warning
    verified = data.get("verified", False)
    review_required = data.get("review_required", True)

    if not verified or review_required:
        notice = (
            "UNVERIFIED - MANUAL REVIEW REQUIRED. Check names, dates, "
            "results, units, and reference ranges against the original "
            "document. This report is not a medical diagnosis."
        )
    else:
        notice = (
            "The source system marks this report as verified. "
            "Follow your organization's review procedures."
        )

    story.append(Paragraph(escape(notice), styles["notice"]))

    # 1. Patient and report details
    story.append(_section("1. Patient & Report Information", styles))

    fields = data.get("fields") or {}
    labels = [
        ("patient_name", "Patient Name"),
        ("hospital_name", "Hospital / Laboratory"),
        ("doctor_name", "Doctor"),
        ("admission_date", "Admission Date"),
        ("discharge_date", "Discharge Date"),
        ("report_id", "Report ID"),
        ("patient_id", "Patient ID"),
        ("lab_id", "Lab ID"),
    ]

    field_rows = [["Field", "Extracted Value"]]

    for key, label in labels:
        if fields.get(key) is not None:
            field_rows.append([label, fields[key]])

    if len(field_rows) == 1:
        field_rows.append(["Patient / Report Details", "No fields detected"])

    story.append(
        _table(field_rows, [55 * mm, 119 * mm], styles)
    )

    # 2. Laboratory results
    story.append(_section("2. Laboratory Results", styles))

    lab_rows = [[
        "Test Name",
        "Result",
        "Unit",
        "Reference Range",
        "Reported Flag",
    ]]

    for item in data.get("lab_results") or []:
        lab_rows.append([
            item.get("test_name"),
            item.get("result"),
            item.get("unit"),
            item.get("reference_range"),
            item.get("reported_flag"),
        ])

    if len(lab_rows) == 1:
        lab_rows.append([
            "No lab results detected", "-", "-", "-", "-"
        ])

    story.append(
        _table(
            lab_rows,
            [43 * mm, 23 * mm, 23 * mm, 43 * mm, 42 * mm],
            styles,
        )
    )

    # 3. Vital signs
    story.append(_section("3. Vital Signs & Measurements", styles))

    vitals = data.get("vitals") or {}
    vital_rows = [["Measurement", "Extracted Value"]]

    for key, value in vitals.items():
        vital_rows.append([key.replace("_", " ").title(), value])

    if len(vital_rows) == 1:
        vital_rows.append(["Vitals", "No values detected"])

    story.append(
        _table(vital_rows, [65 * mm, 109 * mm], styles)
    )

    # 4. Clinical information
    story.append(_section("4. Clinical Information", styles))

    clinical = data.get("clinical_information") or {}
    clinical_rows = [["Category", "Extracted Information"]]

    for key, value in clinical.items():
        clinical_rows.append([key.replace("_", " ").title(), value])

    if len(clinical_rows) == 1:
        clinical_rows.append(["Clinical Information", "Not available"])

    story.append(
        _table(clinical_rows, [50 * mm, 124 * mm], styles)
    )

    # 5. Diagnostic report
    story.append(_section("5. Diagnostic Report Details", styles))

    diagnostics = data.get("diagnostic_reports") or {}
    diagnostic_rows = [["Category", "Extracted Information"]]

    for key, value in diagnostics.items():
        diagnostic_rows.append([key.replace("_", " ").title(), value])

    if len(diagnostic_rows) == 1:
        diagnostic_rows.append(["Diagnostic Report", "Not available"])

    story.append(
        _table(diagnostic_rows, [50 * mm, 124 * mm], styles)
    )

    # 6. Medications
    story.append(_section("6. Medications", styles))

    medications = data.get("medications") or []

    if medications and all(isinstance(item, dict) for item in medications):
        medication_keys = list(dict.fromkeys(
            key
            for item in medications
            for key in item.keys()
        ))

        if medication_keys:
            medication_rows = [[
                key.replace("_", " ").title()
                for key in medication_keys
            ]]

            for item in medications:
                medication_rows.append([
                    item.get(key) for key in medication_keys
                ])

            column_width = 174 * mm / len(medication_keys)
            widths = [column_width] * len(medication_keys)

            story.append(
                _table(medication_rows, widths, styles)
            )
        else:
            story.append(
                Paragraph(
                    "No medication details detected.",
                    styles["body"],
                )
            )

    elif medications:
        for item in medications:
            story.append(
                Paragraph(f"- {_display(item)}", styles["body"])
            )
            story.append(Spacer(1, 3))
    else:
        story.append(
            Paragraph(
                "No medications were extracted from the document.",
                styles["body"],
            )
        )

    # 7. OCR warnings
    story.append(_section("7. OCR Warnings & Review", styles))

    warnings = data.get("warnings") or []

    if warnings:
        for warning in warnings:
            story.append(
                Paragraph(f"- {_display(warning)}", styles["small"])
            )
            story.append(Spacer(1, 2))
    else:
        story.append(
            Paragraph(
                "No extraction warnings were returned.",
                styles["body"],
            )
        )

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "<b>Important:</b> This report contains information extracted "
            "from an uploaded document. Missing values and extraction errors "
            "may occur. Do not use this report alone to make diagnosis or "
            "treatment decisions.",
            styles["small"],
        )
    )

    doc.build(
        story,
        onFirstPage=_footer,
        onLaterPages=_footer,
    )

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
