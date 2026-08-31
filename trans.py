#!/usr/bin/env python3
"""trans - English<->Russian translator that prints and speaks the result.

Direction is auto-detected (Cyrillic script check). Translation runs on a
local llama-server hosting TranslateGemma 4B; speech uses piper TTS.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

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


class TranslateError(RuntimeError):
    """Translation failed; str(exc) is safe to show the user."""


def translate(
    server_url: str,
    text: str,
    direction: str,
    timeout: float = 180.0,
    max_attempts: int = 2,
) -> str:
    """Send the translation prompt to llama-server, return bare translation."""
    payload = json.dumps(
        {
            "messages": [
                {"role": "user", "content": build_prompt(text, direction)}
            ]
        }
    ).encode()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            f"{server_url.rstrip('/')}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode())
            content = body["choices"][0]["message"]["content"].strip()
            if not content:
                raise TranslateError("the model returned an empty translation")
            return content
        except urllib.error.HTTPError as err:
            detail = ""
            try:
                detail = err.read().decode(errors="replace")[:300]
            except Exception:
                pass
            finally:
                err.close()
            raise TranslateError(
                f"translation server returned HTTP {err.code}: {detail}"
            ) from err
        except TranslateError:
            raise
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as err:
            raise TranslateError(
                f"unexpected response from translation server: {err}"
            ) from err
        except (urllib.error.URLError, ConnectionError, TimeoutError) as err:
            last_error = err
            if attempt < max_attempts:
                time.sleep(0.5)
    raise TranslateError(f"cannot reach translation server: {last_error}")
