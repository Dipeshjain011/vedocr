from services.field_extractor import extract_fields

def test_extract_labeled_dates():
    text = """City Hospital
Patient Name: Riya Sharma
Doctor Name: Dr. A. Mehta
Report Date: 12/06/2026
Admission Date: 10/06/2026
Discharge Date: 14/06/2026
"""
    fields, warnings = extract_fields(text)
    assert fields["patient_name"] == "Riya Sharma"
    assert fields["document_date"] == "2026-06-12"
    assert fields["admission_date"] == "2026-06-10"
    assert fields["discharge_date"] == "2026-06-14"
