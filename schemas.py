from typing import Optional, Any
from pydantic import BaseModel, Field


class ExtractedFields(BaseModel):
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None

    hospital_name: Optional[str] = None
    doctor_name: Optional[str] = None
    pathologist_name: Optional[str] = None

    document_type: Optional[str] = None
    report_id: Optional[str] = None
    lab_id: Optional[str] = None

    document_date: Optional[str] = None
    report_date: Optional[str] = None
    order_date: Optional[str] = None
    sample_collection_date: Optional[str] = None
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None

    document_time: Optional[str] = None
    report_time: Optional[str] = None
    order_time: Optional[str] = None
    sample_collection_time: Optional[str] = None


class DocumentTimeline(BaseModel):
    date: Optional[str] = None
    date_source: Optional[str] = None
    confidence: Optional[str] = None


class LabResult(BaseModel):
    test_name: str
    category: Optional[str] = None
    result: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    reported_flag: Optional[str] = None
    status: Optional[str] = None
    sample_type: Optional[str] = None
    collection_date: Optional[str] = None
    method: Optional[str] = None
    notes: Optional[str] = None
    source_text: Optional[str] = None


class Medication(BaseModel):
    medicine_name_and_instructions: str
    strength: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = None


class OCRPage(BaseModel):
    page: int
    engine: str
    status: str
    text: str = ""


class OCRResponse(BaseModel):
    filename: str
    document_type: str
    fields: ExtractedFields
    timeline: DocumentTimeline = Field(default_factory=DocumentTimeline)

    lab_results: list[LabResult] = Field(default_factory=list)
    clinical_information: dict[str, Any] = Field(default_factory=dict)
    medications: list[Medication] = Field(default_factory=list)
    vitals: dict[str, Any] = Field(default_factory=dict)
    diagnostic_reports: dict[str, Any] = Field(default_factory=dict)

    pages: list[OCRPage] = Field(default_factory=list)
    raw_text: str = ""

    review_required: bool = True
    verified: bool = False
    warnings: list[str] = Field(default_factory=list)
