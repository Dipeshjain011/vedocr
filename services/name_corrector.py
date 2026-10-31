
import re


def correct_name(name: str | None) -> str | None:
    """
    Clean OCR name formatting and correct common OCR mistakes.
    """

    if not name or not name.strip():
        return name

    # Remove extra spaces.
    name = re.sub(r"\s+", " ", name).strip()

    # Normalize capitalization:
    # "EMILY JOHNSOn" -> "Emily Johnson"
    name = name.title()

    # Fix selected, common OCR spelling errors.
    # Only add corrections when you have strong evidence.
    corrections = {
        "Johns0n": "Johnson",
        "JohnsOn": "Johnson",
    }

    words = name.split()
    corrected_words = [corrections.get(word, word) for word in words]

    return " ".join(corrected_words)

