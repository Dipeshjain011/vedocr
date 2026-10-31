
from datetime import date


def sort_results(
    results: list[dict],
    sort_by: str = "admission_date",
    order: str = "desc",
) -> list[dict]:
    """
    Sort medical reports by admission date.

    Valid dates are sorted according to the requested order.
    Reports with missing or invalid dates always remain at the end.
    Sorting is stable, so the original order of equal dates is preserved.
    """

    if order not in {"asc", "desc"}:
        raise ValueError("order must be 'asc' or 'desc'")

    if sort_by not in {"admission_date", "discharge_date"}:
        raise ValueError(
            "sort_by must be 'admission_date' or 'discharge_date'"
        )

    dated = []
    undated = []

    for index, item in enumerate(results):
        fields = item.get("fields") or {}
        value = fields.get(sort_by)

        try:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Missing date")

            parsed_date = date.fromisoformat(value.strip())
            dated.append((parsed_date, index, item))

        except (ValueError, TypeError):
            undated.append(item)

    dated.sort(
        key=lambda entry: entry[0],
        reverse=(order == "desc"),
    )

    return [entry[2] for entry in dated] + undated
