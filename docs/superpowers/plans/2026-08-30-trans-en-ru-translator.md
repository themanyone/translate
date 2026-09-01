# trans — English↔Russian Speech Translator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `trans` command that translates typed phrases between English and Russian (auto-detected direction), prints the translation, and speaks it aloud with piper TTS.

**Architecture:** Single-file stdlib Python app talking to a persistent `llama-server` (TranslateGemma 4B GGUF + repo's `translategemma.jinja` template) over its OpenAI-compatible REST API, and shelling out to piper + a WAV player for speech. Direction detection is a Cyrillic-script heuristic. Server lifecycle is user-configurable: auto-start detached by default, or fully manual.

**Tech Stack:** Python 3.14 (stdlib only — `urllib`, `json`, `subprocess`, `argparse`), `llama-server` from llama.cpp 0.3.0-dev, piper TTS with `ru_RU-irina-medium` and `en_US-libritts_r-medium` ONNX voices, `pw-play` (PipeWire) for playback.

## Global Constraints

- **Stdlib only** — no pip dependencies; imports limited to `argparse`, `json`, `os`, `re`, `shutil`, `signal`, `socket`, `subprocess`, `sys`, `time`, `urllib.request`, `urllib.error`, `pathlib`.
- **Single source file** `trans.py` at repo root; installed as `/home/k/.local/bin/trans` via a symlink (no packaging, no venv).
- **Model paths (verbatim):**
  - GGUF: `$HOME/.cache/huggingface/hub/models--mradermacher--translategemma-4b-it-i1-GGUF/snapshots/ffb12df0e4a6d7a4c500376b1d6a66d73409e085/translategemma-4b-it.i1-IQ4_NL.gguf`
  - Chat template: `<repo>/translategemma.jinja`
- **Piper (verbatim):** binary `$HOME/.local/sbin/piper`, voices `$HOME/.cache/piper/ru_RU-irina-medium.onnx` and `$HOME/.cache/piper/en_US-libritts_r-medium.onnx`, always `--cuda`.
- **Server:** `llama-server` (on PATH), default `127.0.0.1:8144`, flags `--no-jinja --chat-template-file translategemma.jinja --temp 0`; state dir `/home/k/.local/state/trans/` (`server.log`, `server.pid`).
- **Prompt format (verbatim)** — user message sent to `/v1/chat/completions`:
  `Translate the following {source_language} text into {target_language}. Produce only the {target_language} translation, without any additional explanations or commentary: {text}`
  where `{source_language}`/`{target_language}` are `English`/`Russian`. This long instruction is what suppresses commentary (verified experimentally; short prompts like `Translate to Russian: …` produce unwanted multi-option explanations).
- **Env vars:** `TRANS_SERVER_URL` (full URL, overrides host/port), `TRANS_AUTO_START` (`1` default, `0` = never auto-start).
- **Temperature:** pinned to 0 server-side only; the client sends no sampling parameters.
- **Platform:** Linux; paths use `pathlib`/`os.path` so the app is not portable to Windows (fine).
- Language detection rule: any character in Unicode range U+0400–U+04FF (Cyrillic) → input is Russian → translate to English; otherwise → translate to Russian. Empty/whitespace-only input is ignored by both CLI entry points.
- Never log or echo secrets; there are none in this app.
- Match repo style: this is a brand-new repo (only `translategemma.jinja` + `docs/`), so style = clean, commented-where-non-obvious, no em dashes in source.

## File Structure

- Create `trans.py` — the entire app (single file per spec; ~300 lines, one responsibility per section: server manager, detector, translator client, speaker, CLI/REPL).
- Create `tests/test_trans.py` — unit tests for pure logic (detection, prompt building, output cleaning, arg parsing, server-command building). Stdlib `unittest` (no pytest in repo conventions yet; keeps "stdlib only" true for tests too). Live end-to-end checks are manual scripts in the plan, not unit tests.
- Create `README.md` — usage, install, env vars, examples.

Task order: detection + prompt building (pure logic, testable without GPU) → translator client → server manager → speaker → CLI/REPL wiring → install + end-to-end → docs.

---

### Task 1: Language detection and prompt building (pure logic)

**Files:**
- Create: `trans.py`
- Create: `tests/test_trans.py`

**Interfaces:**
- Produces:
  - `detect_direction(text: str) -> str | None` — returns `"ru"` if input contains Cyrillic (source Russian, target English), `"en"` otherwise; `None` for empty/whitespace-only input.
  - `build_prompt(text: str, direction: str) -> str` — returns the verbatim-format instruction string. `direction` is `"en"` (English→Russian) or `"ru"` (Russian→English). Raises `ValueError` on other values.
  - `LANG_NAMES: dict[str, tuple[str, str]]` — maps direction code to `(source_language, target_language)` display names: `{"en": ("English", "Russian"), "ru": ("Russian", "English")}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trans.py`:

```python
import unittest

from trans import build_prompt, detect_direction


class TestDetectDirection(unittest.TestCase):
    def test_ascii_input_is_english(self):
        self.assertEqual(detect_direction("Hello! My name is Eek."), "en")

    def test_cyrillic_input_is_russian(self):
        self.assertEqual(detect_direction("Здравствуйте. Меня зовут Иик."), "ru")

    def test_mixed_script_counts_as_russian(self):
        self.assertEqual(detect_direction("call me Иик please"), "ru")

    def test_empty_input_returns_none(self):
        self.assertIsNone(detect_direction(""))

    def test_whitespace_only_returns_none(self):
        self.assertIsNone(detect_direction("   \n\t "))

    def test_empty_after_strip_returns_none(self):
        self.assertIsNone(detect_direction("  "))


class TestBuildPrompt(unittest.TestCase):
    def test_english_to_russian_uses_verbatim_format(self):
        self.assertEqual(
            build_prompt("Hello! My name is Eek.", "en"),
            "Translate the following English text into Russian. "
            "Produce only the Russian translation, without any additional "
            "explanations or commentary: Hello! My name is Eek.",
        )

    def test_russian_to_english_uses_verbatim_format(self):
        self.assertEqual(
            build_prompt("Привет!", "ru"),
            "Translate the following Russian text into English. "
            "Produce only the English translation, without any additional "
            "explanations or commentary: Привет!",
        )

    def test_unknown_direction_raises(self):
        with self.assertRaises(ValueError):
            build_prompt("hi", "fr")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 tests/test_trans.py -v` (from repo root; the test imports `trans` from cwd)
Expected: FAIL with `ModuleNotFoundError: No module named 'trans'`

- [ ] **Step 3: Write minimal implementation**

Create `trans.py`:

```python
#!/usr/bin/env python3
"""trans - English<->Russian translator that prints and speaks the result.

Direction is auto-detected (Cyrillic script check). Translation runs on a
local llama-server hosting TranslateGemma 4B; speech uses piper TTS.
"""

from __future__ import annotations

import re
import sys

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 tests/test_trans.py -v`
Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add trans.py tests/test_trans.py
git commit -m "feat: add language detection and translation prompt building"
```

---

### Task 2: Translation client

**Files:**
- Modify: `trans.py` (append new section)
- Modify: `tests/test_trans.py` (append new test class)

**Interfaces:**
- Consumes: `build_prompt(text, direction)` from Task 1.
- Produces:
  - `translate(server_url: str, text: str, direction: str, timeout: float = 180.0, max_attempts: int = 2) -> str`
    POSTs `{"messages": [{"role": "user", "content": <build_prompt>}]}` to `{server_url}/v1/chat/completions`, returns `choices[0].message.content` stripped. Retries once on `URLError`/`ConnectionError`/`TimeoutError` (not on HTTP status errors). Raises `TranslateError` (subclass of `RuntimeError`) with a human-readable message when the server returns an HTTP error, malformed JSON, or empty content.
  - `TranslateError(RuntimeError)` — exception class; `str(exc)` is user-presentable (no tracebacks in REPL).
- Tests use `unittest.mock.patch` on `urllib.request.urlopen` — no network needed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trans.py` (update the top import line to also import the new names):

```python
import json
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

from trans import TranslateError, build_prompt, detect_direction, translate
```

Then append this test class before the `if __name__ == "__main__":` block:

```python
class TestTranslate(unittest.TestCase):
    SERVER = "http://127.0.0.1:8144"

    def _ok_response(self, content: str) -> mock.MagicMock:
        resp = mock.MagicMock()
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        ).encode()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        return resp

    def _assert_request_body(self, urlopen_mock) -> dict:
        req = urlopen_mock.call_args[0][0]
        self.assertEqual(
            req.full_url, f"{self.SERVER}/v1/chat/completions"
        )
        self.assertEqual(
            req.get_header("Content-type"), "application/json"
        )
        return json.loads(req.data)

    def test_returns_translation_content(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._ok_response("Привет! Меня зовут Ик.")
            result = translate(self.SERVER, "Hello! My name is Eek.", "en")
        self.assertEqual(result, "Привет! Меня зовут Ик.")
        body = self._assert_request_body(up)
        self.assertEqual(
            body["messages"],
            [
                {
                    "role": "user",
                    "content": build_prompt("Hello! My name is Eek.", "en"),
                }
            ],
        )
        self.assertNotIn("temperature", body)

    def test_strips_whitespace_from_content(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._ok_response("  Hello.\n")
            result = translate(self.SERVER, "Привет.", "ru")
        self.assertEqual(result, "Hello.")

    def test_retries_once_on_connection_error_then_succeeds(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.side_effect = [URLError("conn refused"), self._ok_response("Ок.")]
            result = translate(self.SERVER, "OK.", "en")
        self.assertEqual(result, "Ок.")
        self.assertEqual(up.call_count, 2)

    def test_raises_after_exhausting_retries(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.side_effect = URLError("conn refused")
            with self.assertRaises(TranslateError):
                translate(self.SERVER, "OK.", "en")
        self.assertEqual(up.call_count, 2)

    def test_http_error_raises_without_retry(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.side_effect = HTTPError(
                f"{self.SERVER}/v1/chat/completions", 500, "err", {}, None
            )  # type: ignore[arg-type]
            with self.assertRaises(TranslateError):
                translate(self.SERVER, "OK.", "en")
        self.assertEqual(up.call_count, 1)

    def test_malformed_json_raises_translate_error(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            resp = mock.MagicMock()
            resp.read.return_value = b"not json"
            resp.__enter__.return_value = resp
            up.return_value = resp
            with self.assertRaises(TranslateError):
                translate(self.SERVER, "OK.", "en")

    def test_empty_content_raises_translate_error(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._ok_response("   ")
            with self.assertRaises(TranslateError):
                translate(self.SERVER, "OK.", "en")

    def test_missing_choices_raises_translate_error(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            resp = mock.MagicMock()
            resp.read.return_value = json.dumps({"choices": []}).encode()
            resp.__enter__.return_value = resp
            up.return_value = resp
            with self.assertRaises(TranslateError):
                translate(self.SERVER, "OK.", "en")


if __name__ == "__main__":
    unittest.main()
```

(If the file already ends with the `if __name__ == "__main__":` block from Task 1, replace it with this one; keep exactly one such block.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 tests/test_trans.py -v`
Expected: FAIL with `ImportError: cannot import name 'TranslateError'` (or similar)

- [ ] **Step 3: Write minimal implementation**

Append to `trans.py`:

```python
import json
import time
import urllib.error
import urllib.request


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
            raise TranslateError(
                f"translation server returned HTTP {err.code}: {detail}"
            ) from err
        except TranslateError:
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError) as err:
            last_error = err
            if attempt < max_attempts:
                time.sleep(0.5)
    raise TranslateError(f"cannot reach translation server: {last_error}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 tests/test_trans.py -v`
Expected: 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add trans.py tests/test_trans.py
git commit -m "feat: add REST translation client for llama-server"
```

---

### Task 3: Server manager (health check, auto-start, stop)

**Files:**
- Modify: `trans.py` (append new section)
- Modify: `tests/test_trans.py` (append new test class)

**Interfaces:**
- Consumes: none new (stdlib only).
- Produces:
  - `DEFAULT_HOST = "127.0.0.1"`, `DEFAULT_PORT = 8144`
  - `MODEL_PATH = "mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL"`
  - `TEMPLATE_PATH = Path(__file__).resolve().parent / "translategemma.jinja"` — the jinja file sits next to the installed script; when installed via symlink, `__file__` resolves through the symlink to the repo copy (verify in Task 6; if a symlink breaks resolution, `Path(__file__).resolve()` follows it to the real repo file, which is the point).
  - `STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", "/home/k/.local/state")) / "trans"` (expanded).
  - `health_url(host, port) -> str` — `http://{host}:{port}/health`
  - `chat_url(host, port) -> str` — `http://{host}:{port}` (the `translate()` client appends `/v1/chat/completions`)
  - `server_command(host: str, port: int) -> list[str]` — the exact argv for llama-server:
    `["llama-server", "-hf", MODEL_PATH, "--host", host, "--port", str(port), "--no-webui", "--no-jinja", "--chat-template-file", str(TEMPLATE_PATH), "--temp", "0"]`
  - `wait_for_server(url: str, timeout: float = 120.0) -> bool` — poll `GET {url}/health` every 2 s; True once the response body contains `"ok"`.
  - `ensure_server(host: str, port: int, auto_start: bool, stderr=sys.stderr) -> str` — returns the base URL to use. If healthy, return it. Else if `auto_start`, launch detached and wait for health; on success return URL, on timeout print log-path hint to `stderr` and raise `TranslateError`. Else (no auto-start) print the exact `server_command()` shell line to `stderr` and raise `TranslateError`.
  - `stop_server(state_dir: Path = STATE_DIR) -> bool` — SIGTERM the pid in `state_dir/server.pid`, wait up to 10 s for exit, return True/False. Missing pidfile → False.
  - Launch details: `subprocess.Popen(cmd, stdout=open(state_dir/"server.log","ab"), stderr=STDOUT, stdin=DEVNULL, start_new_session=True)`; write `cmd`, pid to `state_dir/server.pid` (first line pid, second line command for humans). `STATE_DIR` created with `mkdir(parents=True, exist_ok=True)`.
- Tests patch `trans.urllib.request.urlopen` and `subprocess.Popen` — no real server spawned.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trans.py` (extend imports at top):

```python
import os
import subprocess
from pathlib import Path

import trans
from trans import (
    TranslateError,
    build_prompt,
    detect_direction,
    ensure_server,
    health_url,
    server_command,
    stop_server,
    translate,
)
```

(Keep `json`, `mock`, `HTTPError`, `URLError`, `unittest` imports from Task 2. Drop the old single-line `from trans import ...` line; this block replaces it.)

Append this class:

```python
class TestServerManager(unittest.TestCase):
    HOST, PORT = "127.0.0.1", 8144

    def _health_ok(self, body=b'{"status":"ok"}') -> mock.MagicMock:
        resp = mock.MagicMock()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        return resp

    def test_urls(self):
        self.assertEqual(health_url("127.0.0.1", 8144), "http://127.0.0.1:8144/health")

    def test_server_command_is_verbatim(self):
        cmd = server_command("127.0.0.1", 8144)
        self.assertEqual(cmd[0], "llama-server")
        self.assertIn("--no-jinja", cmd)
        self.assertIn("--temp", cmd)
        self.assertEqual(cmd[cmd.index("--temp") + 1], "0")
        self.assertEqual(cmd[cmd.index("--chat-template-file") + 1],
                         str(trans.TEMPLATE_PATH))
        self.assertEqual(cmd[cmd.index("-hf") + 1], trans.MODEL_PATH)
        self.assertIn("--port", cmd)

    def test_ensure_server_returns_url_when_healthy(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._health_ok()
            url = ensure_server(self.HOST, self.PORT, auto_start=True)
        self.assertEqual(url, "http://127.0.0.1:8144")

    def test_ensure_server_starts_detached_when_unhealthy(self):
        with (
            mock.patch("trans.urllib.request.urlopen") as up,
            mock.patch("trans.subprocess.Popen") as popen,
            mock.patch("trans.wait_for_server", return_value=True) as wait,
            mock.patch("trans.STATE_DIR", Path("/tmp/fake-state")),
        ):
            up.side_effect = URLError("refused")
            popen.return_value.pid = 4242
            url = ensure_server(self.HOST, self.PORT, auto_start=True)
        self.assertEqual(url, "http://127.0.0.1:8144")
        self.assertEqual(popen.call_count, 1)
        argv = popen.call_args[0][0]
        self.assertEqual(argv[0], "llama-server")
        popen_kwargs = popen.call_args[1]
        self.assertTrue(popen_kwargs["start_new_session"])
        wait.assert_called_once_with("http://127.0.0.1:8144")
        pidfile = Path("/tmp/fake-state") / "server.pid"
        self.assertEqual(pidfile.read_text().splitlines()[0], "4242")

    def test_ensure_server_raises_when_start_times_out(self):
        with (
            mock.patch("trans.urllib.request.urlopen") as up,
            mock.patch("trans.subprocess.Popen") as popen,
            mock.patch("trans.wait_for_server", return_value=False),
            mock.patch("trans.STATE_DIR", Path("/tmp/fake-state")),
        ):
            up.side_effect = URLError("refused")
            popen.return_value.pid = 99
            with self.assertRaises(TranslateError), \
                    mock.patch("sys.stderr") as err:
                ensure_server(self.HOST, self.PORT, auto_start=True)
        self.assertIn("server.log", "".join(str(c) for c in err.write.call_args))

    def test_ensure_server_manual_mode_prints_command_and_raises(self):
        with mock.patch("trans.urllib.request.urlopen") as up, \
                mock.patch("sys.stderr") as err:
            up.side_effect = URLError("refused")
            with self.assertRaises(TranslateError):
                ensure_server(self.HOST, self.PORT, auto_start=False)
        written = "".join(str(c) for c in err.write.call_args)
        self.assertIn("llama-server", written)
        self.assertIn("8144", written)

    def test_stop_server_sigterms_pid(self):
        with mock.patch("trans.subprocess.Popen") as popen, \
                mock.patch("pathlib.Path.exists", return_value=True), \
                mock.patch("builtins.open", mock.mock_open(
                    read_data="1234\nllama-server -m x")), \
                mock.patch("trans.os.kill") as kill, \
                mock.patch("trans.time.time") as now:
            popen.return_value.poll.side_effect = [None, None, None, None,
                                                   None, 0]
            now.side_effect = [0, 2, 4, 6, 8, 10]
            self.assertTrue(stop_server(Path("/tmp/fake-state")))
        kill.assert_called_once()
        self.assertEqual(kill.call_args[0][0], 1234)

    def test_stop_server_missing_pidfile_returns_false(self):
        with mock.patch("pathlib.Path.exists", return_value=False):
            self.assertFalse(stop_server(Path("/tmp/fake-state")))


if __name__ == "__main__":
    unittest.main()
```

(Again keep a single trailing `if __name__ == "__main__":` block.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 tests/test_trans.py -v`
Expected: FAIL with `ImportError: cannot import name 'ensure_server'` (or similar)

- [ ] **Step 3: Write minimal implementation**

Append to `trans.py`:

```python
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8144

MODEL_PATH = "mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL"

TEMPLATE_PATH = Path(__file__).resolve().parent / "translategemma.jinja"

STATE_DIR = (Path(os.environ.get("XDG_STATE_HOME", "/home/k/.local/state")) / "trans").expanduser()


def health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/health"


def chat_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def server_command(host: str, port: int) -> list[str]:
    return [
        "llama-server",
        "-hf", MODEL_PATH,
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
    stderr: object = None,
) -> str:
    """Return a healthy server base URL, starting the server if allowed."""
    out = stderr if stderr is not None else sys.stderr
    url = chat_url(host, port)
    if _server_is_healthy(health_url(host, port)):
        return url
    if not auto_start:
        cmd_line = " ".join(shlex_quote(part) for part in server_command(host, port))
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
```

Add `import shlex` to the imports at the top and define this tiny helper next to the manager section:

```python
def shlex_quote(part: str) -> str:
    return shlex.quote(part)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 tests/test_trans.py -v`
Expected: 23 tests PASS
Note: if `test_stop_server_sigterms_pid` fails because of how mocks interact with `open`, fix the mock wiring in the test (not the implementation) — the production code path is straightforward.

- [ ] **Step 5: Commit**

```bash
git add trans.py tests/test_trans.py
git commit -m "feat: add llama-server lifecycle management"
```

---

### Task 4: Speaker (piper TTS + playback)

**Files:**
- Modify: `trans.py` (append new section)
- Modify: `tests/test_trans.py` (append new test class)

**Interfaces:**
- Consumes: direction codes `"en"`/`"ru"` (Task 1 vocabulary).
- Produces:
  - `PIPER = "/home/k/.local/sbin/piper"`
  - `VOICES = {"ru": os.path.join(os.environ.get("HOME", "/home/user"), ".cache/piper/ru_RU-irina-medium.onnx"), "en": os.path.join(os.environ.get("HOME", "/home/user"), ".cache/piper/en_US-libritts_r-medium.onnx")}`
  - `PLAYER_CANDIDATES = ["pw-play", "paplay", "aplay"]`
  - `find_player() -> str | None` — first candidate on PATH via `shutil.which`, else None.
  - `speak(text: str, direction_out: str) -> bool` — synthesizes with the voice matching the output language (for direction `"en"` the output is Russian → use `VOICES["ru"]`; for `"ru"` → `VOICES["en"]`), pipes WAV through the player, returns True if the pipeline succeeded. Implementation: `p1 = Popen([PIPER, "--cuda", "--model", voice, "-f", "-"], stdin=PIPE, stdout=PIPE); p2 = Popen([player, "-"], stdin=p1.stdout)` — feed text to piper's stdin, close handles, wait for both; False if either exits non-zero or player is None. Never raises (prints a warning to stderr on failure).
- Tests patch `subprocess.Popen` and `shutil.which` — no audio played.

- [ ] **Step 1: Write the failing tests**

Extend `tests/test_trans.py` imports:

```python
from trans import find_player, speak
```

Append:

```python
class TestSpeaker(unittest.TestCase):
    def test_find_player_first_on_path(self):
        with mock.patch("trans.shutil.which", side_effect=lambda n: f"/usr/bin/{n}"):
            self.assertEqual(find_player(), "/usr/bin/pw-play")

    def test_find_player_none_when_missing(self):
        with mock.patch("trans.shutil.which", return_value=None):
            self.assertIsNone(find_player())

    def test_speak_russian_voice_for_english_direction(self):
        # direction "en" = English input, so the spoken translation is Russian
        with mock.patch("trans.find_player", return_value="pw-play"), \
                mock.patch("trans.subprocess.Popen") as popen:
            popen.side_effect = [
                mock.MagicMock(stdout=mock.MagicMock()),  # piper
                mock.MagicMock(),                          # player
            ]
            for p in popen.side_effect:
                p.poll.return_value = 0
            self.assertTrue(speak("Привет!", "en"))
        piper_argv = popen.call_args_list[0][0][0]
        self.assertEqual(piper_argv[0], os.path.join(os.environ.get("HOME", "/home/user"), ".local/sbin/piper"))
        self.assertIn("--cuda", piper_argv)
        self.assertEqual(piper_argv[piper_argv.index("--model") + 1],
                         os.path.join(os.environ.get("HOME", "/home/user"), ".cache/piper/ru_RU-irina-medium.onnx"))
        self.assertEqual(piper_argv[piper_argv.index("-f") + 1], "-")

    def test_speak_english_voice_for_russian_direction(self):
        with mock.patch("trans.find_player", return_value="pw-play"), \
                mock.patch("trans.subprocess.Popen") as popen:
            popen.side_effect = [
                mock.MagicMock(stdout=mock.MagicMock()),
                mock.MagicMock(),
            ]
            for p in popen.side_effect:
                p.poll.return_value = 0
            self.assertTrue(speak("Hello!", "ru"))
        piper_argv = popen.call_args_list[0][0][0]
        self.assertEqual(piper_argv[piper_argv.index("--model") + 1],
                         os.path.join(os.environ.get("HOME", "/home/user"), ".cache/piper/en_US-libritts_r-medium.onnx"))

    def test_speak_returns_false_without_player(self):
        with mock.patch("trans.find_player", return_value=None):
            self.assertFalse(speak("hi", "en"))

    def test_speak_returns_false_when_piper_fails(self):
        with mock.patch("trans.find_player", return_value="pw-play"), \
                mock.patch("trans.subprocess.Popen") as popen:
            bad_piper = mock.MagicMock(stdout=mock.MagicMock())
            bad_piper.poll.return_value = 1
            popen.return_value = [bad_piper, mock.MagicMock()]
            popen.side_effect = None
            with mock.patch("sys.stderr"):
                self.assertFalse(speak("hi", "en"))


if __name__ == "__main__":
    unittest.main()
```

(Keep a single trailing `if __name__ == "__main__":` block. Note: `Popen` is called with `stdin=piper.stdin` style chaining for the player; the test only asserts the first call's argv. In `test_speak_returns_false_when_piper_fails`, `popen.return_value`/`side_effect` juggling is only there so a broken first process surfaces; if the implementation never reaches the player call because piper failed immediately, make the test assert `False` and not care how many Popen calls happened.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 tests/test_trans.py -v`
Expected: FAIL with `ImportError: cannot import name 'find_player'`

- [ ] **Step 3: Write minimal implementation**

Append to `trans.py`:

```python
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
        print("warning: no audio player found (pw-play, paplay, aplay)", file=sys.stderr)
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
        assert piper_proc.stdin is not None and piper_proc.stdout is not None
        piper_proc.stdin.write(text.encode())
        piper_proc.stdin.close()
        piper_proc.stdout.close()
        piper_rc = piper_proc.wait()
        player_rc = player_proc.wait()
        if piper_rc != 0 or player_rc != 0:
            print(f"warning: speech pipeline failed (piper={piper_rc}, player={player_rc})",
                  file=sys.stderr)
            return False
        return True
    except OSError as err:
        print(f"warning: speech failed: {err}", file=sys.stderr)
        return False
```

(`shutil` was already imported in Task 3.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 tests/test_trans.py -v`
Expected: 28 tests PASS
If mock of `Popen` makes `speak` take a wrong branch, fix the test's mock side effects — the production logic is final.

- [ ] **Step 5: Commit**

```bash
git add trans.py tests/test_trans.py
git commit -m "feat: add piper TTS speech output"
```

---

### Task 5: CLI and REPL wiring

**Files:**
- Modify: `trans.py` (append new section)
- Modify: `tests/test_trans.py` (append new test class)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces:
  - `parse_args(argv: list[str]) -> argparse.Namespace` with fields: `phrase: list[str]` (possibly empty), `speak_flag: bool` (`--no-speak` sets False), `host: str` (`--host`, default `DEFAULT_HOST`), `port: int` (`--port`, default `DEFAULT_PORT`), `stop: bool` (`--stop-server`), plus `--help`/`--version` (version string `trans 1.0.0`).
  - `resolve_server_url(args, env: dict[str, str]) -> tuple[str, bool]` — returns `(url, auto_start)`. `TRANS_SERVER_URL` env (non-empty) overrides everything and forces `auto_start=False` (external server, never manage it). Otherwise `(chat_url(host, port), env.get("TRANS_AUTO_START", "1") != "0")`.
  - `translate_once(text: str, server_url: str, do_speak: bool, stdout=sys.stdout) -> None` — detect direction, call `translate`, print `ru: <text>` or `en: <text>` (the prefix is the OUTPUT language code), then speak if requested. Detection-None (blank input) prints nothing.
  - `run_repl(server_url: str, do_speak: bool, stdin=sys.stdin, stdout=sys.stdout) -> None` — print usage hint lines, loop `input()`; `q`/`exit`/EOF returns; KeyboardInterrupt during translation returns to a fresh prompt; per-line errors from `TranslateError` are printed and the loop continues.
  - `main(argv: list[str] | None = None) -> int` — exit code 0 on success, 1 on fatal errors; `--stop-server` handled before server-connection logic; never raises (catches `TranslateError`, `OSError` at top).
  - `if __name__ == "__main__": sys.exit(main())` at file bottom.
- Printing rule: translations always go to stdout with prefix `ru: ` or `en: ` (output language); status/warnings to stderr — so stdout stays pipeable.

- [ ] **Step 1: Write the failing tests**

Extend imports in `tests/test_trans.py`:

```python
from trans import main, parse_args, resolve_server_url, translate_once, run_repl
```

Append:

```python
class TestCLI(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = parse_args(["hello", "world"])
        self.assertEqual(args.phrase, ["hello", "world"])
        self.assertTrue(args.speak_flag)
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8144)
        self.assertFalse(args.stop)

    def test_parse_args_no_speak(self):
        self.assertFalse(parse_args(["--no-speak", "hi"]).speak_flag)

    def test_parse_args_port(self):
        self.assertEqual(parse_args(["--port", "9000", "hi"]).port, 9000)

    def test_parse_args_stop_server(self):
        self.assertTrue(parse_args(["--stop-server"]).stop)

    def test_parse_args_empty_phrase_is_repl(self):
        self.assertEqual(parse_args([]).phrase, [])

    def test_parse_args_rejects_unknown_flag(self):
        with self.assertRaises(SystemExit), \
                mock.patch("sys.stderr"):
            parse_args(["--bogus", "hi"])

    def test_resolve_url_env_override_disables_autostart(self):
        url, auto = resolve_server_url(
            parse_args(["--port", "1234", "hi"]),
            {"TRANS_SERVER_URL": "http://elsewhere:9999"},
        )
        self.assertEqual(url, "http://elsewhere:9999")
        self.assertFalse(auto)

    def test_resolve_url_defaults(self):
        url, auto = resolve_server_url(parse_args(["hi"]), {})
        self.assertEqual(url, "http://127.0.0.1:8144")
        self.assertTrue(auto)

    def test_resolve_url_autostart_env_off(self):
        url, auto = resolve_server_url(parse_args(["hi"]), {"TRANS_AUTO_START": "0"})
        self.assertFalse(auto)

    def test_translate_once_prints_output_language_prefix(self):
        out = io.StringIO()
        with mock.patch("trans.translate", return_value="Привет!"), \
                mock.patch("trans.speak", return_value=True) as sp:
            translate_once("Hello!", "http://x", speak_flag=True, stdout=out)
        self.assertEqual(out.getvalue(), "ru: Привет!\n")
        sp.assert_called_once_with("Привет!", "en")

    def test_translate_once_no_speak(self):
        out = io.StringIO()
        with mock.patch("trans.translate", return_value="Hello."), \
                mock.patch("trans.speak") as sp:
            translate_once("Привет.", "http://x", speak_flag=False, stdout=out)
        self.assertEqual(out.getvalue(), "en: Hello.\n")
        sp.assert_not_called()

    def test_translate_once_blank_input_noop(self):
        out = io.StringIO()
        with mock.patch("trans.translate") as tr:
            translate_once("   ", "http://x", speak_flag=True, stdout=out)
        tr.assert_not_called()
        self.assertEqual(out.getvalue(), "")

    def test_run_repl_translates_lines_until_quit(self):
        out = io.StringIO()
        fake_input = iter(["Hello!", "   ", "q"]).__next__
        with mock.patch("trans.translate", return_value="Привет!"), \
                mock.patch("trans.speak", return_value=True), \
                mock.patch("builtins.input", side_effect=["Hello!", "   ", "q"]):
            run_repl("http://x", speak_flag=True, stdout=out)
        self.assertEqual(out.getvalue().count("ru: Привет!\n"), 1)

    def test_run_repl_survives_translate_error(self):
        out = io.StringIO()
        with mock.patch("trans.translate", side_effect=TranslateError("boom")), \
                mock.patch("trans.speak"), \
                mock.patch("builtins.input", side_effect=["Hello!", "q"]), \
                mock.patch("sys.stderr"):
            run_repl("http://x", speak_flag=True, stdout=out)

    def test_main_stop_server(self):
        with mock.patch("trans.stop_server", return_value=True) as stop, \
                mock.patch("sys.stdout"):
            code = main(["--stop-server"])
        self.assertEqual(code, 0)
        stop.assert_called_once()

    def test_main_one_shot(self):
        with mock.patch("trans.ensure_server", return_value="http://x"), \
                mock.patch("trans.translate", return_value="Привет!"), \
                mock.patch("trans.speak", return_value=True), \
                mock.patch("sys.stdout"):
            code = main(["Hello!"])
        self.assertEqual(code, 0)

    def test_main_one_shot_error_exits_nonzero(self):
        with mock.patch("trans.ensure_server", return_value="http://x"), \
                mock.patch("trans.translate", side_effect=TranslateError("bad")), \
                mock.patch("sys.stderr"), \
                mock.patch("sys.stdout"):
            self.assertEqual(main(["Hello!"]), 1)

    def test_main_fatal_error_returns_one(self):
        with mock.patch("trans.ensure_server",
                        side_effect=TranslateError("no server")), \
                mock.patch("sys.stderr"):
            self.assertEqual(main(["Hello!"]), 1)


if __name__ == "__main__":
    unittest.main()
```

Also add `import io` to the test file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 tests/test_trans.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_args'`

- [ ] **Step 3: Write minimal implementation**

Append to `trans.py`:

```python
import argparse

__version__ = "1.0.0"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trans",
        description="Translate between English and Russian, print and speak it.",
    )
    parser.add_argument("phrase", nargs="*", help="phrase to translate (omit for interactive mode)")
    parser.add_argument("--no-speak", dest="speak_flag", action="store_false",
                        help="print the translation without speaking it")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"server host (default {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"server port (default {DEFAULT_PORT})")
    parser.add_argument("--stop-server", action="store_true",
                        help="stop the auto-started llama-server and exit")
    parser.add_argument("--version", action="version", version=f"trans {__version__}")
    return parser.parse_args(argv)


def resolve_server_url(args: argparse.Namespace, env: dict[str, str]) -> tuple[str, bool]:
    """Figure out which server to use and whether we may auto-start it."""
    external = env.get("TRANS_SERVER_URL", "").strip()
    if external:
        # External server: never try to manage its lifecycle.
        return external.rstrip("/"), False
    auto_start = env.get("TRANS_AUTO_START", "1").strip() != "0"
    return chat_url(args.host, args.port), auto_start


def translate_once(
    text: str,
    server_url: str,
    speak_flag: bool,
    stdout: object = None,
) -> None:
    out = stdout if stdout is not None else sys.stdout
    direction = detect_direction(text)
    if direction is None:
        return
    translation = translate(server_url, text, direction)
    print(f"{LANG_NAMES[direction][1][:2].lower()}: {translation}", file=out)
    if speak_flag:
        speak(translation, direction)


def run_repl(server_url: str, speak_flag: bool, stdout: object = None) -> None:
    out = stdout if stdout is not None else sys.stdout
    print("type a phrase in English or Russian; q to quit", file=out)
    while True:
        try:
            line = input()
        except EOFError:
            return
        except KeyboardInterrupt:
            print(file=out)
            return
        if line.strip().lower() in {"q", "exit", "quit"}:
            return
        try:
            translate_once(line, server_url, speak_flag, stdout=out)
        except TranslateError as err:
            print(f"error: {err}", file=sys.stderr)
        except KeyboardInterrupt:
            print("(interrupted)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.stop:
        if stop_server():
            print("server stopped")
            return 0
        print("no recorded server to stop", file=sys.stderr)
        return 1
    server_url, auto_start = resolve_server_url(args, dict(os.environ))
    try:
        server_url = ensure_server(args.host, args.port, auto_start)
    except TranslateError:
        return 1
    try:
        if args.phrase:
            translate_once(" ".join(args.phrase), server_url, args.speak_flag)
        else:
            run_repl(server_url, args.speak_flag)
    except TranslateError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except OSError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Notes for the implementer:
- In `translate_once` the printed prefix is the OUTPUT language: `LANG_NAMES[direction][1]` is the target name, `[:2].lower()` gives `ru`/`en`.
- `run_repl`'s inner `KeyboardInterrupt` handler exists for Ctrl+C during a long translation: it prints and loops back to `input()`.
- `input()` prompts on the tty directly; the REPL prints no extra prompt string.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 tests/test_trans.py -v`
Expected: 46 tests PASS

- [ ] **Step 5: Commit**

```bash
git add trans.py tests/test_trans.py
git commit -m "feat: add CLI and interactive REPL"
```

---

### Task 6: Install and end-to-end verification (live GPU)

**Files:**
- Create: `README.md`
- Create symlink: `/home/k/.local/bin/trans` -> `/home/k/.local/src/python/trans/trans.py` (not committed; do not gitignore-hack it, it lives outside the repo)

**Interfaces:**
- Consumes: full app.
- Produces: installed `trans` command; verified end-to-end behavior.

- [ ] **Step 1: Install the command**

```bash
chmod +x trans.py
mkdir -p /home/k/.local/bin
ln -sf /home/k/.local/src/python/trans/trans.py /home/k/.local/bin/trans
hash -r; which trans
```

Expected: `/home/k/.local/bin/trans` (ensure `/home/k/.local/bin` precedes any stale `trans` in PATH).

- [ ] **Step 2: Verify unit suite still passes via installed path**

Run: `python3 /home/k/.local/bin/trans --version`
Expected: `trans 1.0.0`
Run: `python3 tests/test_trans.py -v`
Expected: 46 PASS
Also verify `Path(__file__).resolve()` follows the symlink so the installed command finds `translategemma.jinja`:
`python3 -c "from trans import TEMPLATE_PATH; print(TEMPLATE_PATH); print(TEMPLATE_PATH.exists())"` (from repo root)
Expected: `/home/k/.local/src/python/trans/translategemma.jinja`, `True`

- [ ] **Step 3: Live test - server already running (en→ru, speak)**

If a server is already listening on 8144, kill it first so auto-start is exercised: `pkill -f 'llama-server.*8144'` (only if you own the process).
Run: `timeout 300 trans Hello! My name is Eek.`
Expected: stderr shows "starting llama-server..."; after load (~20-60 s first time), stdout shows `ru: Здравствуйте! Меня зовут Ик.` (capitalization/transliteration of "Eek" may vary slightly, e.g. `Ик`); audio plays.

- [ ] **Step 4: Live test - second run is fast (ru→en, speak)**

Run: `timeout 120 trans Здравствуйте. Меня зовут Иик.`
Expected: within a few seconds, stdout `en: Hello. My name is Iik.`; audio plays. This verifies both direction detection (Cyrillic input) and warm-server reuse (no "starting" message).

- [ ] **Step 5: Live test - one-shot with --no-speak and manual mode**

Run: `timeout 60 trans --no-speak The weather is beautiful today.`
Expected: prints `ru: ...` only, no audio, quick.
Run: `TRANS_AUTO_START=0 timeout 10 trans test one two`
Expected (with server stopped): prints the exact `llama-server` launch command containing `--port 8144` and exits 1; no audio.
Run: `trans --stop-server`
Expected: `server stopped`, and `ss -ltn | grep 8144` shows nothing.
Run: `TRANS_AUTO_START=0 timeout 10 trans test three`
Expected: same manual-mode failure message (proves stopping really stopped it).

- [ ] **Step 6: Live test - REPL**

Run: `timeout 120 trans` with stdin lines `Hello, how are you?` then `Хорошо, спасибо.` then `q`.
Expected: two translations printed and spoken, then clean exit 0. Test Ctrl+C at the prompt as well: second invocation, press Ctrl+C at the empty prompt; expected clean exit (code 130 from wrapper or 0; either accepted, note which).

- [ ] **Step 7: Write README.md**

Create `README.md`:

```markdown
# trans

English <-> Russian translator that prints the translation and speaks it
with piper TTS. Language direction is detected automatically from the script
of what you type (Cyrillic -> English, otherwise -> Russian).

## Requirements

- llama.cpp's `llama-server` (on PATH)
- TranslateGemma 4B GGUF (already at the hardcoded model path)
- piper TTS at `/home/k/.local/sbin/piper` with the `ru_RU-irina-medium` and
  `en_US-libritts_r-medium` voices
- a WAV player: `pw-play`, `paplay`, or `aplay`

## Install

```sh
ln -sf "$PWD/trans.py" /home/k/.local/bin/trans
```

## Usage

- `trans` - interactive mode; type phrases, `q` quits.
- `trans <phrase...>` - translate one phrase and exit.
- `trans --no-speak <phrase...>` - print only.
- `trans --stop-server` - stop the auto-started llama-server.

The app talks to `llama-server` at `127.0.0.1:8144` (override with
`--host`/`--port`). By default it auto-starts a detached server if none is
running and leaves it running afterwards. Server logs live in
`/home/k/.local/state/trans/server.log`.

Set `TRANS_AUTO_START=0` to manage the server yourself: the app then prints
the exact command it wants you to run and exits if the server is missing.
Set `TRANS_SERVER_URL=http://host:port` to use an external server entirely.

## Development

```sh
python3 tests/test_trans.py -v
```
```

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs: add usage and install instructions"
```

---

## Self-Review Notes (verified during planning)

- Spec coverage: detection (Task 1), prompt format verbatim (Task 1), REST client + retries + timeouts (Task 2), server lifecycle incl. env vars and manual mode (Task 3), piper speech + player fallback (Task 4), REPL + one-shot + flags + exit codes (Task 5), install + live verification of every spec behavior (Task 6), README (Task 6).
- No placeholders: every step has full code or an exact command with expected output.
- Type consistency: `detect_direction -> "en"|"ru"|None`, `translate(server_url, text, direction)`, `ensure_server(host, port, auto_start) -> url`, `speak(text, direction)`, `translate_once(text, server_url, speak_flag, stdout=None)`, `run_repl(server_url, speak_flag, stdout=None)`, `parse_args(argv) -> Namespace` with `phrase`, `speak_flag`, `host`, `port`, `stop`, `resolve_server_url(args, env) -> (url, auto_start)`, `main(argv=None) -> int`. Direction codes are consistently `"en"` = English input, `"ru"` = Russian input.
- Deliberate deviation from TDD skill's usual pytest: repo has no pytest config; stdlib `unittest` honors the "stdlib only" constraint and runs via `python3 tests/test_trans.py -v`.
