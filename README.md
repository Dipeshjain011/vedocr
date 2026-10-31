# Medical Reports OCR Backend
### Lab Reports & Medical Reports Processing | VedAmrit

A backend-only Medical Optical Character Recognition (OCR) service built using **Python and FastAPI**. It extracts, processes, and organizes information from medical reports and laboratory documents into structured JSON responses.

This service is developed as the Medical OCR component of **VedAmrit**, helping convert scattered medical documents into organized digital information.

> **Project Scope:** This repository provides backend OCR functionality only. It does not include a frontend, database, or handwritten-prescription processing pipeline.

---

## 1. Overview

Medical reports are commonly available as digital PDFs, scanned PDFs, and images. Extracting important information manually from these documents can be time-consuming.

This project automates the process by:
- Accepting medical reports and laboratory documents.
- Extracting embedded PDF text.
- Performing OCR on scanned documents and images.
- Identifying important medical information.
- Extracting laboratory results.
- Normalizing dates into a consistent format.
- Sorting multiple documents chronologically.
- Returning structured JSON responses.

## 2. Extracted Information

The backend extracts the following information whenever it can be confidently identified:

| Field | Description |
|---|---|
| Patient Name | Name of the patient |
| Hospital Name | Hospital or laboratory name |
| Doctor Name | Doctor or referring doctor name |
| Document Date | Date associated with the report |
| Admission Date | Patient admission date |
| Discharge Date | Patient discharge date |
| Document Type | Type of medical document |
| Raw OCR Text | Text extracted from the document |
| Extraction Warnings | Information about extraction issues |
| Review Flag | Indicates whether manual verification is required |

### Date Handling

- Dates are normalized into ISO format: `YYYY-MM-DD`, when confidently recognized.
- Original date text is preserved.
- Documents with unknown dates are placed last during sorting.
- Default chronological sorting uses the report/document date.
- Admission and discharge dates are maintained as separate fields.

---

## 3. Key Features

### PDF and Image Processing
- Supports medical PDF documents and images.
- Extracts embedded text from searchable PDFs.
- Processes scanned documents through OCR when the required OCR engine is configured.

### OCR Processing
- Uses PyMuPDF for PDF text extraction and page processing.
- Uses PaddleOCR for OCR on scanned documents and images.
- Supports processing through a modular OCR service.

### Medical Field Extraction
- Extracts patient, hospital, and doctor information.
- Identifies report, admission, and discharge dates.
- Identifies the document type.
- Preserves raw extracted text for further verification.

### Laboratory Results Extraction
- Extracts laboratory test information from supported reports.
- Organizes extracted test names, values, and units.
- Provides structured information for further processing.

### Chronological Sorting
- Sorts multiple medical documents by report/document date.
- Uses ascending order: oldest report first.
- Places documents with unknown dates at the end.

### Structured API Responses
- Returns extracted information in JSON format.
- Includes extraction warnings and review indicators.
- Supports integration with other backend applications.

---

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend REST API framework |
| Uvicorn | ASGI server for running the application |
| PyMuPDF (fitz) | PDF text extraction and page processing |
| PaddleOCR | OCR engine for scanned documents and images |
| PaddlePaddle | Runtime framework required by PaddleOCR |
| OpenCV | Image processing and enhancement |
| Pillow | Image loading and manipulation |
| NumPy | Numerical and image-array operations |
| Regular Expressions | Pattern-based extraction of names, dates, and medical fields |
| Pydantic | Data validation and structured API schemas |
| JSON | Structured API response format |
| Pytest | Automated testing |

---

## 5. System Architecture

The backend follows a modular document-processing workflow.

```text
             Medical PDF / Image
                      |
                      v
                FastAPI Backend
                      |
                      v
               OCR API Router
                      |
                      v
               Document Loader
                 (PyMuPDF)
                      |
             +--------+--------+
             |                 |
             v                 v
       Embedded Text       Scanned Pages
       Extraction                |
                                v
                          Image Processing
                                |
                                v
                            PaddleOCR
                                |
             +------------------+
             |
             v
        Extracted Text
             |
             v
       Field Extraction
             |
       +-----+------+
       |            |
       v            v
 Medical Fields  Lab Results
       |            |
       +------+-----+
              |
              v
       Date Normalization
              |
              v
      Chronological Sorting
              |
              v
       Structured JSON
              |
              v
       API Response
```

---

## 6. Project Structure

The backend is organized into separate modules to keep document loading, OCR processing, field extraction, and API handling independent.

```text
vedocr/
│
├── main.py
├── config.py
├── schemas.py
├── requirements.txt
│
├── routers/
│   └── ocr.py
│
├── services/
│   ├── document_loader.py
│   ├── printed_ocr.py
│   ├── ocr_service.py
│   ├── field_extractor.py
│   ├── lab_results_extractor.py
│   ├── sorting.py
│   └── name_corrector.py
│
└── tests/
    ├── test_field_extractor.py
    └── test_sorting.py
```

*This represents the main modules of the backend. Additional files and test-data folders may be present in the repository.*

---

## 7. Installation and Setup

### Prerequisites

- Python 3.11 (recommended)
- pip
- Git
- PaddleOCR
- Compatible PaddlePaddle runtime

### Step 1: Clone the Repository

```bash
git clone https://github.com/Dipeshjain011/vedocr.git
```

Navigate to the project directory:

```bash
cd vedocr
```

### Step 2: Create a Virtual Environment

```powershell
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Install PaddleOCR

Install PaddleOCR and its compatible PaddlePaddle runtime separately by following the official installation guide:

https://www.paddleocr.ai/main/en/version3.x/installation.html

Make sure the installed PaddleOCR and PaddlePaddle versions are compatible with your Python environment.

---

## 8. Running the Backend

Start the FastAPI application from the project root:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The backend will be available at:

**Base URL:**

http://127.0.0.1:8000

**Interactive API Documentation:**

http://127.0.0.1:8000/docs

**Alternative API Documentation:**

http://127.0.0.1:8000/redoc

**Health Check:**

http://127.0.0.1:8000/health

---

## 9. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks backend health |
| POST | `/api/v1/ocr` | Processes one PDF or image |
| POST | `/api/v1/ocr/batch` | Processes multiple PDF or image documents |

### Supported Document Types

Both OCR endpoints support the following `document_type` values:

| Value | Description |
|---|---|
| `auto` | Automatically determines the document type |
| `lab_report` | Processes laboratory reports |
| `medical_report` | Processes general medical reports |

### Batch Processing

The batch endpoint:
- Accepts multiple medical documents.
- Processes the uploaded documents.
- Returns extracted results.
- Sorts results by report/document date in ascending order.
- Places documents with unknown dates last.

---

## 10. Example API Request

The following Python example demonstrates how to upload a medical report using the OCR API.

```python
import requests

with open("report.pdf", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/ocr",
        files={
            "file": ("report.pdf", f, "application/pdf")
        },
        data={
            "document_type": "auto"
        },
        timeout=180,
    )

response.raise_for_status()

print(response.json())
```

---

## 11. Example JSON Response

The following is an illustrative example of a structured OCR response:

```json
{
  "filename": "medical_report.pdf",
  "document_type": "lab_report",
  "fields": {
    "patient_name": "Maya Patel",
    "hospital_name": "Riverbend Demo Clinic",
    "doctor_name": "Dr. Rohan Shah",
    "report_date": "2026-10-22",
    "admission_date": "2026-10-21",
    "discharge_date": "2026-10-23"
  },
  "review_required": true,
  "warnings": [],
  "raw_text": "Extracted medical report text..."
}
```

*The example is illustrative. Actual response fields and values depend on the document, OCR quality, and extraction results.*

---

## 12. Testing

The project includes tests for important extraction and sorting functionality.

Run the available test suite:

```bash
python -m pytest tests/ -v
```

Run field extraction tests:

```bash
python -m pytest tests/test_field_extractor.py -v
```

Run sorting tests:

```bash
python -m pytest tests/test_sorting.py -v
```

---

## 13. Important Limitations

- Field extraction uses transparent heuristics on OCR text, not a guaranteed medical NLP model.
- Medical document layouts vary, so extraction patterns may need refinement using representative, de-identified documents.
- If PaddleOCR is not installed, searchable PDFs can still be processed through embedded text extraction.
- Scanned images require the appropriate PaddleOCR setup. Without it, the backend returns a clear setup status.
- Names and dates may be misread due to scan quality, document layout, or OCR limitations.
- Extracted information must always be compared with the original medical document.
- Keep `review_required=true` until extracted information has been verified.
- This backend does not provide medical diagnosis or replace professional medical judgment.

---

## 14. Security and Privacy Considerations

This API does not persist patient files.

Before production deployment, configure:

- Authentication and authorization.
- Access controls for medical documents.
- Encryption for data in transit and at rest wherever data is stored.
- Secure file handling and retention policies.
- Secure logging that avoids exposing sensitive patient information.
- Appropriate privacy and healthcare compliance requirements.

---

## 15. Future Enhancements

Potential improvements for future versions include:

- Improved OCR accuracy across different hospital report layouts.
- Better extraction of complex laboratory tables.
- Improved recognition of patient and doctor names.
- Enhanced date extraction and validation.
- Confidence scoring and human verification workflows.
- Multilingual medical document support.
- Integration with secure medical record storage.
- Improved OCR processing speed and scalability.

---

## 16. VedAmrit Integration

VedOCR is designed as the Medical OCR component of the **VedAmrit** healthcare platform.

It helps transform scattered medical reports into structured digital information that can support:

- Organized patient medical history.
- Digital medical document management.
- Chronological report organization.
- Access to extracted laboratory information.
- Integration with other VedAmrit healthcare modules.

---

## 17. Repository

**GitHub Repository:**  
https://github.com/Dipeshjain011/vedocr

---

## 18. Author

**Dipesh Jain**  
Developer – VedOCR Medical OCR Backend  
Project – VedAmrit

---

## License

No license is specified in this README. Add an appropriate license file if you intend to distribute or reuse this project publicly.
