from services.sorting import sort_results

def test_dates_ascending_unknown_last():
    items = [
        {"filename": "later", "fields": {"document_date": "2026-08-20"}},
        {"filename": "unknown", "fields": {"document_date": None}},
        {"filename": "earlier", "fields": {"document_date": "2026-06-12"}},
    ]
    result = sort_results(items)
    assert [x["filename"] for x in result] == ["earlier", "later", "unknown"]
