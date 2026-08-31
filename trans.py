#!/usr/bin/env python3
"""trans - English<->Russian translator that prints and speaks the result.

Direction is auto-detected (Cyrillic script check). Translation runs on a
local llama-server hosting TranslateGemma 4B; speech uses piper TTS.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

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


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8144

MODEL_PATH = (
    "/home/k/.cache/huggingface/hub/models--mradermacher--translategemma-4b-it-i1-GGUF/"
    "snapshots/ffb12df0e4a6d7a4c500376b1d6a66d73409e085/"
    "translategemma-4b-it.i1-IQ4_NL.gguf"
)

TEMPLATE_PATH = Path(__file__).resolve().parent / "translategemma.jinja"

STATE_DIR = (
    Path(os.environ.get("XDG_STATE_HOME", "~/.local/state")) / "trans"
).expanduser()


def health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/health"


def chat_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def server_command(host: str, port: int) -> list[str]:
    return [
        "llama-server",
        "-m", MODEL_PATH,
        "--host", host,
        "--port", str(port),
        "--no-webui",
        "--no-jinja",
        "--chat-template-file", str(TEMPLATE_PATH),
        "--temp", "0",
    ]


def _server_is_healthy(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return b"ok" in response.read().lower()
    except (urllib.error.URLError, OSError):
        return False


def wait_for_server(url: str, timeout: float = 120.0) -> bool:
    """Poll the health endpoint until it answers or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_is_healthy(url):
            return True
        time.sleep(2.0)
    return False


def _start_server_detached(host: str, port: int, state_dir: Path) -> int:
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "server.log"
    cmd = server_command(host, port)
    with open(log_path, "ab") as log_file:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (state_dir / "server.pid").write_text(f"{process.pid}\n{' '.join(cmd)}\n")
    return process.pid


def ensure_server(
    host: str,
    port: int,
    auto_start: bool,
    stderr=None,
) -> str:
    """Return a healthy server base URL, starting the server if allowed."""
    out = stderr if stderr is not None else sys.stderr
    url = chat_url(host, port)
    if _server_is_healthy(health_url(host, port)):
        return url
    if not auto_start:
        cmd_line = " ".join(shlex.quote(part) for part in server_command(host, port))
        print(
            f"translation server is not running at {url}\n"
            f"start it with:\n  {cmd_line}",
            file=out,
        )
        raise TranslateError("translation server unavailable")
    print("starting llama-server (first load takes a while)...", file=out)
    _start_server_detached(host, port, STATE_DIR)
    if not wait_for_server(health_url(host, port)):
        print(
            f"server did not become healthy; see {STATE_DIR / 'server.log'}",
            file=out,
        )
        raise TranslateError("translation server failed to start")
    return url


def stop_server(state_dir: Path | None = None) -> bool:
    """SIGTERM the server recorded in state_dir/server.pid; True if stopped."""
    state_dir = state_dir if state_dir is not None else STATE_DIR
    pid_file = state_dir / "server.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().splitlines()[0])
        os.kill(pid, signal.SIGTERM)
    except (ValueError, ProcessLookupError, PermissionError):
        return False
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.5)
    return False


PIPER = "/home/k/.local/sbin/piper"

# Output language -> piper voice. Direction "en" means English was typed,
# so the translation (and speech) is Russian.
VOICES: dict[str, str] = {
    "ru": "/home/k/.cache/piper/ru_RU-irina-medium.onnx",
    "en": "/home/k/.cache/piper/en_US-libritts_r-medium.onnx",
}

PLAYER_CANDIDATES = ["pw-play", "paplay", "aplay"]


def find_player() -> str | None:
    for candidate in PLAYER_CANDIDATES:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def speak(text: str, direction: str) -> bool:
    """Say the translation aloud; False on any failure (never raises)."""
    voice = VOICES["ru" if direction == "en" else "en"]
    player = find_player()
    if player is None:
        print(
            "warning: no audio player found (pw-play, paplay, aplay)",
            file=sys.stderr,
        )
        return False
    try:
        piper_proc = subprocess.Popen(
            [PIPER, "--cuda", "--model", voice, "-f", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        player_proc = subprocess.Popen(
            [player, "-"],
            stdin=piper_proc.stdout,
        )
        piper_proc.stdin.write(text.encode())
        piper_proc.stdin.close()
        piper_proc.stdout.close()
        piper_rc = piper_proc.wait()
        player_rc = player_proc.wait()
        if piper_rc != 0 or player_rc != 0:
            print(
                f"warning: speech pipeline failed "
                f"(piper={piper_rc}, player={player_rc})",
                file=sys.stderr,
            )
            return False
        return True
    except OSError as err:
        print(f"warning: speech failed: {err}", file=sys.stderr)
        return False
