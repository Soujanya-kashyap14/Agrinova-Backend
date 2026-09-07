"""Detect which language a farmer actually used, from the script of
the text itself - used to override a stale/default UI language toggle
so the assistant replies in the language the farmer really spoke or
typed, not whatever the language selector happened to be set to."""

_SCRIPT_RANGES = {
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "hi": (0x0900, 0x097F),  # Devanagari (Hindi)
    "te": (0x0C00, 0x0C7F),  # Telugu
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "ml": (0x0D00, 0x0D7F),  # Malayalam
}


def detect_script_language(text: str) -> str | None:
    """Return the language code whose script dominates `text`, or
    None if the text is plain Latin/ASCII (no non-English script
    characters at all, so there is no reliable signal here).

    A farmer's Kannada speech, once transcribed to Kannada script, IS
    Kannada regardless of what the UI's language dropdown says - the
    dropdown may be left on a previous choice, or default to English,
    while the farmer naturally speaks or types in their own language.
    """

    counts = {code: 0 for code in _SCRIPT_RANGES}

    for ch in text:
        codepoint = ord(ch)
        for code, (start, end) in _SCRIPT_RANGES.items():
            if start <= codepoint <= end:
                counts[code] += 1
                break

    best_code, best_count = max(counts.items(), key=lambda item: item[1])

    return best_code if best_count > 0 else None
