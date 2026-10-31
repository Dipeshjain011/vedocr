# Medical Reports OCR Backend (Lab + Medical Reports)

Backend-only FastAPI service. No frontend, no database, and no handwritten-prescription pipeline.

## Extracted fields
- patient name
- hospital / laboratory name
- doctor name
- document/report date
- admission date
- discharge date
- document type
- raw OCR text
- extraction warnings and review flag

Dates are normalized to ISO `YYYY-MM-DD` when confidently recognized. Original date text is preserved. Documents with unknown dates are sorted last. The default chronology uses the report/document date, not admission date. Admission and discharge dates remain separate fields.

## Run on Windows
Use Python 3.11 if installed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PaddleOCR and its compatible PaddlePaddle runtime separately by following:
https://www.paddleocr.ai/main/en/version3.x/installation.html

Start:
```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
API docs: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/health

## Endpoints
- `GET /health`
- `POST /api/v1/ocr` — one PDF/image
- `POST /api/v1/ocr/batch` — multiple PDF/images; returns results sorted by report date ascending (oldest first), unknown dates last

For both OCR endpoints, `document_type` can be `auto`, `lab_report`, or `medical_report`.

## Example client request
```python
import requests

with open("report.pdf", "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/api/v1/ocr",
        files={"file": ("report.pdf", f, "application/pdf")},
        data={"document_type": "auto"},
        timeout=180,
    )
r.raise_for_status()
print(r.json())
```

## Important limitations
- Field extraction uses transparent heuristics on OCR text, not a guaranteed medical NLP model. Real-world layouts vary, so expect to refine labels and patterns using representative, de-identified examples.
- If PaddleOCR is not installed, searchable PDFs still work through embedded text extraction; scanned images return a clear setup status.
- Names and dates can be misread. Always compare extracted fields with the original and keep `review_required=true` until verified.
- No patient files are persisted by this API. Configure authentication, access controls, encryption, retention, and secure logging before production use.
