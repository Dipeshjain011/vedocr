
$responseFile = ".\phase5_scanned_response.json"

if (-not (Test-Path $responseFile)) {
    Write-Host "ERROR: Response JSON not found." -ForegroundColor Red
    exit 1
}

$data = Get-Content $responseFile -Raw | ConvertFrom-Json
$tests = @()

function Add-Test {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Details
    )

    $script:tests += [PSCustomObject]@{
        Test    = $Name
        Result  = if ($Passed) { "PASS" } else { "FAIL" }
        Details = $Details
    }
}

Add-Test "Correct filename" `
    ($data.filename -eq "06_scanned_image_based_report.pdf") `
    "Actual: $($data.filename)"

Add-Test "One page detected" `
    (@($data.pages).Count -eq 1) `
    "Actual pages: $(@($data.pages).Count)"

$expectedFields = @(
    @{ Name = "Patient name";   Key = "patient_name";   Expected = "Emily Johnson" },
    @{ Name = "Doctor name";    Key = "doctor_name";    Expected = "Taylor Newton" },
    @{ Name = "Hospital name";  Key = "hospital_name";  Expected = "Nunez-Humphrey" },
    @{ Name = "Admission date"; Key = "admission_date"; Expected = "2023-12-20" },
    @{ Name = "Discharge date"; Key = "discharge_date"; Expected = "2023-12-24" },
    @{ Name = "Report ID";      Key = "report_id";      Expected = "OCR-SCAN-TEST-006" },
    @{ Name = "Patient ID";     Key = "patient_id";     Expected = "OCR-PAT-TEST-006" }
)

foreach ($field in $expectedFields) {
    $actual = [string]$data.fields.($field.Key)

    Add-Test $field.Name `
        ($actual -eq $field.Expected) `
        "Expected: $($field.Expected); Actual: $actual"
}

$medicationFound = $false
foreach ($med in $data.medications) {
    if (($med | ConvertTo-Json -Compress -Depth 10) -match '(?i)Ibuprofen') {
        $medicationFound = $true
        break
    }
}

Add-Test "Medication: Ibuprofen" $medicationFound "Medication extraction"

$engineFound = $false
foreach ($page in $data.pages) {
    if ([string]$page.engine -match '(?i)tesseract') {
        $engineFound = $true
        break
    }
}

Add-Test "Tesseract engine reported" $engineFound "OCR engine metadata"
Add-Test "Review required" ([bool]$data.review_required) "Human review required"
Add-Test "Not marked verified" (-not [bool]$data.verified) "Unverified OCR status"

$rawText = [string]$data.raw_text
if ([string]::IsNullOrWhiteSpace($rawText)) {
    $rawText = ($data.pages | ForEach-Object { [string]$_.text }) -join "`n"
}

Add-Test "Patient label in OCR text" ($rawText -match '(?i)Patient Name') "Source text"
Add-Test "Asthma in OCR text" ($rawText -match '(?i)Asthma') "Source text"
Add-Test "Report ID in OCR text" ($rawText -match '(?i)OCR-SCAN-TEST-006') "Source text"

Write-Host "`n===== SCANNED PDF REGRESSION TEST =====" -ForegroundColor Cyan
$tests | Format-Table -AutoSize -Wrap

$total = $tests.Count
$passed = @($tests | Where-Object Result -eq "PASS").Count
$failed = $total - $passed
$rate = if ($total -gt 0) { [math]::Round(100 * $passed / $total, 2) } else { 0 }

Write-Host "`nTotal checks : $total"
Write-Host "Passed       : $passed" -ForegroundColor Green
Write-Host "Failed       : $failed" -ForegroundColor $(if ($failed) { "Red" } else { "Green" })
Write-Host "Pass rate    : $rate%"

Write-Host "`nDocument type (informational): $($data.document_type)"
Write-Host "Lab results (informational): $(@($data.lab_results).Count)"

$tests | Export-Csv ".\phase5_scanned_test_results.csv" -NoTypeInformation
Write-Host "`nResults saved to phase5_scanned_test_results.csv"
