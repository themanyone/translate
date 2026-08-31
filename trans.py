#!/usr/bin/env python3
"""trans - English<->Russian translator that prints and speaks the result.

Direction is auto-detected (Cyrillic script check). Translation runs on a
local llama-server hosting TranslateGemma 4B; speech uses piper TTS.
"""

from __future__ import annotations

import re

# Direction code -> (source language name, target language name)
LANG_NAMES: dict[str, tuple[str, str]] = {
    "en": ("English", "Russian"),
    "ru": ("Russian", "English"),
}

# Any character in the Cyrillic block (plus extensions) marks Russian input.
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")


def detect_direction(text: str) -> str | None:
    """Return "en" (English input) or "ru" (Russian input); None if empty."""
    if not text.strip():
        return None
    return "ru" if _CYRILLIC.search(text) else "en"


def build_prompt(text: str, direction: str) -> str:
    """Build the instruction prompt that yields a bare translation.

    The long form matters: short prompts make the model offer multiple
    translation options with commentary instead of just translating.
    """
    try:
        source_name, target_name = LANG_NAMES[direction]
    except KeyError:
        raise ValueError(f"unknown direction: {direction!r}") from None
    return (
        f"Translate the following {source_name} text into {target_name}. "
        f"Produce only the {target_name} translation, without any additional "
        f"explanations or commentary: {text.strip()}"
    )
