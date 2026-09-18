#!/usr/bin/env python3
"""translate - translator that prints and speaks the result.

Translation runs on llama-server. A router-mode server that is already
running and hosts a TranslateGemma model (discovered via /v1/models) is
preferred; requests then select that model by id. Otherwise translate uses
a dedicated single-model server it can start itself. Input language is
detected with the running translation model (or forced with --from).
Output defaults to English; English input defaults to Spanish instead;
--to overrides. Speech uses piper TTS, which plays the audio itself.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Output language when --to is not given: DEFAULT_TARGET for any input,
# SECONDARY_TARGET when the input is English (so it never translates
# en->en).
DEFAULT_TARGET = "en"
SECONDARY_TARGET = "es"

# Dedicated single-model server (the fallback) and its model files.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8144

# Preferred backend: a router-mode llama-server already running and
# hosting a TranslateGemma model (advertised on /v1/models, selected by
# id on every request).
ROUTER_HOST = "127.0.0.1"
ROUTER_PORT = 8087

MODEL_PATH = "mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL"

TEMPLATE_PATH = Path(__file__).resolve().parent / "translategemma.jinja"

HOME = os.environ.get("HOME", os.environ.get("USERPROFILE", "C:\\Users\\user"))

XDG_STATE_FALLBACK = os.path.join(HOME, ".local", "state")
STATE_DIR = (
    Path(os.environ.get("XDG_STATE_HOME", XDG_STATE_FALLBACK)) / "translate"
).expanduser()

# Speech: piper TTS. The binary plays the synthesized audio itself, so no
# output file or separate player is needed.
PIPER = "piper"
PIPER_DIR = Path(HOME) / ".cache/piper"
#PIPER_DIR = Path(os.environ.get("HOME", "/home/user")) / ".cache/piper"

DOWNLOAD_VOICES = "download_voices"
PREFERRED_VOICES: dict[str, str] = {
    "en": "en_US-libritts_r-medium",
    "ru": "ru_RU-irina-medium",
}

# Voices with more than one speaker, from the official piper catalog
# (voices.json). Used only when a speaker id is requested: most preferred
# voices are single-speaker, so -s/-si would otherwise be a silent no-op.
MULTI_SPEAKER_VOICES: dict[str, str] = {
    "de": "de_DE-mls-medium",
    "fr": "fr_FR-mls-medium",
    "nl": "nl_NL-mls-medium",
    "uk": "uk_UA-ukrainian_tts-medium",
    "es": "es_ES-sharvard-medium",
    "vi": "vi_VN-vivos-x_low",
    "ne": "ne_NP-google-medium",
    "bn": "bn_BD-google-medium",
    "no": "no_NO-nvcc-medium",
    "mr": "mr_IN-google-medium",
    "cy": "cy_GB-bu_tts-medium",
    "kk": "kk_KZ-issai-high",
    "et": "et_EE-news-medium",
    "ja": "ja_JP-hi_fi_captain-medium",
    "ku": "ku_TR-berfin_renas-medium",
    "sr": "sr_RS-serbski_institut-medium",
}

# Prompt used to identify the input language via the running model.
DETECT_PROMPT = (
    "Answer with exactly one word: the ISO 639-1 code "
    "(like fr, de, ru, es) of the language of this text. Text: "
)

# ISO 639-1 code -> language name, used to build the translation prompt.
# Keys include common aliases (ISO 639-2/B three-letter codes) so model
# answers like "eng" or "rus" resolve too.
LANGUAGES: dict[str, str] = {
    "en": "English", "eng": "English",
    "ru": "Russian", "rus": "Russian",
    "es": "Spanish", "spa": "Spanish",
    "fr": "French", "fre": "French", "fra": "French",
    "de": "German", "ger": "German", "deu": "German",
    "it": "Italian", "ita": "Italian",
    "pt": "Portuguese", "por": "Portuguese",
    "uk": "Ukrainian", "ukr": "Ukrainian",
    "pl": "Polish", "pol": "Polish",
    "nl": "Dutch", "nld": "Dutch", "dut": "Dutch",
    "sv": "Swedish", "swe": "Swedish",
    "da": "Danish", "dan": "Danish",
    "fi": "Finnish", "fin": "Finnish",
    "cs": "Czech", "ces": "Czech", "cze": "Czech",
    "bg": "Bulgarian", "bul": "Bulgarian",
    "ar": "Arabic", "ara": "Arabic",
    "he": "Hebrew", "heb": "Hebrew",
    "ja": "Japanese", "jpn": "Japanese",
    "zh": "Chinese", "zho": "Chinese", "chi": "Chinese",
    "ko": "Korean", "kor": "Korean",
    "hi": "Hindi", "hin": "Hindi",
    "tr": "Turkish", "tur": "Turkish",
    "el": "Greek", "ell": "Greek", "gre": "Greek",
    "ro": "Romanian", "ron": "Romanian", "rum": "Romanian",
    "hu": "Hungarian", "hun": "Hungarian",
    "no": "Norwegian", "nor": "Norwegian",
    "id": "Indonesian", "ind": "Indonesian",
    "fa": "Persian", "fas": "Persian", "per": "Persian",
}

_ANSWER_RE = re.compile(r"^[^A-Za-z]*([A-Za-z]{2,3})[^A-Za-z]*$")

__version__ = "1.2.0"

# ---------------------------------------------------------------------------


class TranslateError(RuntimeError):
    """Translation failed; str(exc) is safe to show the user."""


def _debug_command(
    cmd: list[str] | str,
    debug: bool,
    stderr=None,
    input_text: str | None = None,
) -> None:
    """Echo the exact shell command being executed when --debug is on."""
    if not debug:
        return
    out = stderr if stderr is not None else sys.stderr
    if isinstance(cmd, str):
        rendered = cmd
    else:
        rendered = " ".join(shlex.quote(part) for part in cmd)
    if input_text is not None:
        rendered += f" <<< {shlex.quote(input_text)}"
    print(f"debug: {rendered}", file=out)


def resolve_language(spec: str) -> str:
    """Map a language name, 639-1 code, or 639-2 alias to a 639-1 code."""
    key = spec.strip().lower()
    if not key:
        raise ValueError("empty language spec")
    if key in LANGUAGES:
        code = key
    else:
        matches = [k for k, v in LANGUAGES.items()
                   if v.lower() == key and len(k) == 2]
        if not matches:
            raise ValueError(f"unknown language: {spec!r}")
        code = matches[0]
    if len(code) != 2:
        code = next(k for k, v in LANGUAGES.items()
                    if v == LANGUAGES[code] and len(k) == 2)
    return code


def pick_target(source: str, to_lang: str | None) -> str:
    """Choose the output language: explicit --to, else English, except
    English input flips to Spanish."""
    if to_lang is not None:
        return to_lang
    return SECONDARY_TARGET if source == "en" else DEFAULT_TARGET


def build_prompt(text: str, source: str, target: str) -> str:
    """Build the instruction prompt that yields a bare translation.

    The long form matters: short prompts make the model offer multiple
    translation options with commentary instead of just translating.
    """
    try:
        source_name = LANGUAGES[source]
        target_name = LANGUAGES[target]
    except KeyError:
        raise ValueError(
            f"unsupported language pair: {source}->{target}"
        ) from None
    return (
        f"Translate the following {source_name} text into {target_name}. "
        f"Produce only the {target_name} translation, without any additional "
        f"explanations or commentary: {text.strip()}"
    )


def _chat(
    server_url: str,
    prompt: str,
    timeout: float = 180.0,
    debug: bool = False,
    stderr=None,
    model: str | None = None,
) -> str:
    """One chat completion; returns the stripped assistant content.

    model names the model to use on a router-mode server; None (the
    dedicated single-model server case) omits the field entirely.
    """
    request_body: dict = {"messages": [{"role": "user", "content": prompt}]}
    if model is not None:
        request_body["model"] = model
    payload = json.dumps(request_body).encode()

    endpoint = f"{server_url.rstrip('/')}/v1/chat/completions"
    _debug_command(
        f"POST {endpoint}", debug, stderr, input_text=prompt
    )
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    return body["choices"][0]["message"]["content"].strip()


def _chat_with_retry(
    server_url: str,
    prompt: str,
    timeout: float = 180.0,
    max_attempts: int = 2,
    debug: bool = False,
    stderr=None,
    model: str | None = None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _chat(server_url, prompt, timeout, debug, stderr, model)
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


def detect_language(
    server_url: str,
    text: str,
    debug: bool = False,
    stderr=None,
    model: str | None = None,
) -> str:
    """Ask the running model for the input text's ISO 639-1 code."""
    if not text.strip():
        raise TranslateError("nothing to detect in empty input")
    answer = _chat_with_retry(
        server_url, DETECT_PROMPT + text.strip(),
        debug=debug, stderr=stderr, model=model,
    )
    match = _ANSWER_RE.match(answer)
    if not match:
        raise TranslateError(f"could not detect language: {answer!r}")
    try:
        return resolve_language(match.group(1))
    except ValueError as err:
        raise TranslateError(f"could not detect language: {answer!r}") from err


def translate(
    server_url: str,
    text: str,
    source: str,
    target: str,
    timeout: float = 180.0,
    max_attempts: int = 2,
    debug: bool = False,
    stderr=None,
    model: str | None = None,
) -> str:
    """Send the translation prompt to llama-server, return bare translation."""
    content = _chat_with_retry(
        server_url,
        build_prompt(text, source, target),
        timeout,
        max_attempts,
        debug,
        stderr,
        model,
    )
    if not content:
        raise TranslateError("the model returned an empty translation")
    return content


def find_router_model(
    server_url: str,
    timeout: float = 5.0,
    debug: bool = False,
    stderr=None,
) -> str | None:
    """Return the id of the TranslateGemma model a server advertises.

    Probes /v1/models, where a router-mode llama-server lists every
    model it can host. Matches the id containing "translategemma", else
    an alias of exactly "translate"; None means this server is unusable for
    translation (unreachable, or no TranslateGemma among its models).
    """
    endpoint = f"{server_url.rstrip('/')}/v1/models"
    _debug_command(f"GET {endpoint}", debug, stderr)
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    entries = body.get("data") or []
    for entry in entries:
        if "translategemma" in str(entry.get("id", "")).lower():
            return entry["id"]
    for entry in entries:
        aliases = [str(alias).lower() for alias in entry.get("aliases") or []]
        if "translate" in aliases or "translategemma" in aliases:
            return entry["id"]
    return None


# --- server manager --------------------------------------------------------

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


def _start_server_detached(
    host: str,
    port: int,
    state_dir: Path,
    debug: bool = False,
    stderr=None,
) -> int:
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "server.log"
    cmd = server_command(host, port)
    _debug_command(cmd, debug, stderr)
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


def _acquire_start_lock(state_dir: Path) -> int | None:
    """Hold an exclusive lock for the check-then-spawn window.

    Returns the open lock fd, or None if another translate instance is
    already mid-start (we then wait for the server, not the lock).
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(state_dir / "start.lock", os.O_CREAT | os.O_RDWR)
    except OSError:
        return None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except OSError:
        os.close(lock_fd)  # contention: close, let the other instance win
        return None


def ensure_server(
    host: str,
    port: int,
    auto_start: bool,
    stderr=None,
    debug: bool = False,
) -> str:
    """Return a healthy server base URL, starting the server if allowed."""
    out = stderr if stderr is not None else sys.stderr
    url = chat_url(host, port)
    if _server_is_healthy(health_url(host, port)):
        return url
    if not auto_start:
        cmd_line = " ".join(
            shlex.quote(part) for part in server_command(host, port)
        )
        print(
            f"translation server is not running at {url}\n"
            f"start it with:\n  {cmd_line}",
            file=out,
        )
        raise TranslateError("translation server unavailable")
    print("starting llama-server (first load takes a while)...", file=out)
    lock_fd = _acquire_start_lock(STATE_DIR)
    try:
        # Re-check under the lock: a racing instance may have just
        # started a healthy server (or our earlier check may be stale).
        if _server_is_healthy(health_url(host, port)):
            return url
        if lock_fd is not None:
            # We hold the lock: we are the instance that spawns the server.
            _start_server_detached(
                host, port, STATE_DIR, debug=debug, stderr=out
            )
        # Either we spawned it or another instance is starting it now;
        # in both cases, wait for the health endpoint.
        if not wait_for_server(health_url(host, port)):
            print(
                f"server did not become healthy; "
                f"see {STATE_DIR / 'server.log'}",
                file=out,
            )
            raise TranslateError("translation server failed to start")
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
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
    except (ValueError, ProcessLookupError):
        # Recorded process is gone (crash/reboot): clear the stale record.
        try:
            pid_file.unlink()
        except OSError:
            pass
        return False
    except PermissionError:
        # Process exists but belongs to another user; leave the record.
        return False
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                pid_file.unlink()
            except OSError:
                pass
            return True
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)  # stubborn server: force
    except (ProcessLookupError, PermissionError):
        pass  # died between probe and SIGKILL
    finally:
        try:
            pid_file.unlink()
        except OSError:
            pass
    return True


# --- speech ----------------------------------------------------------------

def _voice_num_speakers(voice: str) -> int:
    """Number of speakers in a piper voice; 1 when the config is unreadable."""
    try:
        with open(f"{voice}.json", encoding="utf-8") as handle:
            config = json.load(handle)
        return int(config.get("num_speakers", 1))
    except (OSError, ValueError, TypeError):
        return 1


def _download_voice(name: str, debug: bool = False, stderr=None) -> str | None:
    """Download one piper voice; return its .onnx path, or None on failure."""
    print(f"downloading voice {name}...", file=sys.stderr)
    download_cmd = [DOWNLOAD_VOICES, "--download-dir", str(PIPER_DIR), name]
    _debug_command(download_cmd, debug, stderr)
    try:
        result = subprocess.run(
            download_cmd, capture_output=True, text=True, timeout=600
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        print(f"warning: voice download failed: {err}", file=sys.stderr)
        return None
    path = PIPER_DIR / f"{name}.onnx"
    if result.returncode == 0 and path.exists():
        return str(path)
    return None


def voice_for_language(
    lang: str, debug: bool = False, stderr=None, multi_speaker: bool = False
) -> str | None:
    """Find a piper voice file for an output language.

    Preferred voices first, then any {lang}_* file already downloaded,
    then download_voices listing + download of the first match. With
    multi_speaker, only a voice with several speakers is chosen, because
    piper ignores --speaker on single-speaker voices.
    """
    known = MULTI_SPEAKER_VOICES.get(lang)
    preferred = []
    if multi_speaker and known:
        preferred.append(known)
    if PREFERRED_VOICES.get(lang):
        preferred.append(PREFERRED_VOICES[lang])
    for name in preferred:
        path = PIPER_DIR / f"{name}.onnx"
        if not path.exists():
            continue
        if multi_speaker and _voice_num_speakers(str(path)) <= 1:
            continue
        return str(path)
    existing = sorted(PIPER_DIR.glob(f"{lang}_*.onnx"))
    if multi_speaker:
        existing = [p for p in existing if _voice_num_speakers(str(p)) > 1]
    if existing:
        return str(existing[0])
    if multi_speaker:
        if known:
            downloaded = _download_voice(known, debug=debug, stderr=stderr)
            if downloaded is not None:
                return downloaded
        # No multi-speaker voice available: fall back so speech still works
        # (the caller warns that the speaker id has no effect).
        return voice_for_language(lang, debug=debug, stderr=stderr)
    # Not on disk: consult download_voices for the first matching voice.
    _debug_command([DOWNLOAD_VOICES], debug, stderr)
    try:
        listing = subprocess.run(
            [DOWNLOAD_VOICES], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        print(f"warning: cannot list voices: {err}", file=sys.stderr)
        return None
    if listing.returncode != 0:
        return None
    for line in listing.stdout.splitlines():
        name = line.strip()
        if name.startswith(f"{lang}_"):
            return _download_voice(name, debug=debug, stderr=stderr)
    return None


def speak(
    text: str, lang: str, debug: bool = False, stderr=None,
    speaker: int | None = None,
) -> bool:
    """Say the translation aloud; False on any failure (never raises).

    piper plays the audio itself: text goes to its stdin, no output file.
    ``speaker`` selects a voice speaker id; single-speaker voices ignore
    it, so the user is warned instead of being silently ignored.
    """
    out = stderr if stderr is not None else sys.stderr
    voice = voice_for_language(lang, debug=debug, stderr=stderr,
                               multi_speaker=speaker is not None)
    if voice is None:
        print(
            f"warning: no piper voice found for language {lang!r}",
            file=out,
        )
        return False
    piper_cmd = [PIPER, "--cuda", "--model", voice]
    if speaker is not None:
        if _voice_num_speakers(voice) > 1:
            piper_cmd += ["--speaker", str(speaker)]
        else:
            print(
                f"warning: voice {Path(voice).stem} for {lang!r} has one "
                f"speaker, so speaker {speaker} has no effect",
                file=out,
            )
    _debug_command(piper_cmd, debug, stderr, input_text=text)
    # piper chatters onnxruntime/CUDA warnings to stderr; only show them
    # under --debug, where the user is actually diagnosing something.
    piper_stderr = None if debug else subprocess.DEVNULL
    try:
        piper_proc = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stderr=piper_stderr,
        )
        piper_proc.stdin.write(text.encode())
        piper_proc.stdin.close()
        piper_rc = piper_proc.wait()
        if piper_rc != 0:
            print(
                f"warning: speech pipeline failed (piper={piper_rc})",
                file=sys.stderr,
            )
            return False
        return True
    except OSError as err:
        print(f"warning: speech failed: {err}", file=sys.stderr)
        return False


# --- CLI --------------------------------------------------------------------

_SPEAK_INPUT_RE = re.compile(r"^(?:-si|--speak-input)(?:=?([0-9]+))?$")


def _extract_speak_input(argv: list[str]) -> tuple[list[str], int | None]:
    """Pull -si/--speak-input out of argv, capturing an optional speaker id.

    The speaker id may be attached (-si2, -si=2, --speak-input=2) or
    separate (-si 2). A numeric token after the bare flag is the input
    speaker; anything else stays a positional phrase, so bare
    --speak-input keeps working when a phrase follows it.
    """
    rest: list[str] = []
    input_speaker: int | None = None
    i = 0
    while i < len(argv):
        match = _SPEAK_INPUT_RE.match(argv[i])
        if match is None:
            rest.append(argv[i])
            i += 1
            continue
        value = match.group(1)
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if value is None and nxt is not None and re.fullmatch(r"[0-9]+", nxt):
            value = nxt
            i += 2
        else:
            i += 1
        rest.append("--speak-input")
        if value is not None:
            input_speaker = int(value)
    return rest, input_speaker


def parse_args(argv: list[str]) -> argparse.Namespace:
    argv, input_speaker = _extract_speak_input(list(argv))
    parser = argparse.ArgumentParser(
        prog="translate",
        description="Translate text, print and speak it.",
    )
    parser.add_argument(
        "phrase", nargs="*",
        help="phrase to translate (omit for interactive mode)",
    )
    parser.add_argument(
        "--from", dest="from_lang", metavar="LANG",
        help="input language (name or code); default: auto-detect",
    )
    parser.add_argument(
        "--to", dest="to_lang", metavar="LANG",
        help="output language (name or code); default: English, "
        "or Spanish when the input is English",
    )
    parser.add_argument(
        "--no-speak", dest="speak_flag", action="store_false",
        help="print the translation without speaking it",
    )
    parser.add_argument(
        "-si", "--speak-input", dest="speak_input_flag", action="store_true",
        help="speak the input text in its original language (if detected "
        "or specified); -si N also picks a piper speaker id for the input",
    )
    parser.add_argument(
        "-s", "--speaker", type=int, metavar="N",
        help="piper speaker id for multi-speaker voices "
        "(default: the voice's own default)",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"fallback dedicated server host (default {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"fallback dedicated server port (default {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--router-host", default=ROUTER_HOST,
        help=f"router-mode server host (default {ROUTER_HOST})",
    )
    parser.add_argument(
        "--router-port", type=int, default=ROUTER_PORT,
        help=f"router-mode server port (default {ROUTER_PORT})",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="print the exact shell commands and requests being executed",
    )
    parser.add_argument(
        "--stop-server", dest="stop", action="store_true",
        help="stop the auto-started llama-server and exit",
    )
    parser.add_argument(
        "--version", action="version", version=f"translate {__version__}"
    )
    args = parser.parse_args(argv)
    args.input_speaker = input_speaker
    try:
        if args.from_lang is not None:
            args.from_lang = resolve_language(args.from_lang)
        if args.to_lang is not None:
            args.to_lang = resolve_language(args.to_lang)
    except ValueError as err:
        parser.error(str(err))
    return args


def resolve_backend(
    args: argparse.Namespace,
    env: dict[str, str],
    debug: bool = False,
    stderr=None,
) -> tuple[str, str | None]:
    """Return (server_url, model_id) to send translate requests to.

    Prefers a router-mode llama-server already running at
    args.router_host:args.router_port that advertises a TranslateGemma
    model; every request then selects that model by id. Otherwise falls
    back to the dedicated single-model server, auto-starting it when
    TRANS_AUTO_START allows. TRANS_SERVER_URL pins one fixed server
    whose lifecycle translate never manages; model selection still applies
    when that server hosts TranslateGemma.
    """
    out = stderr if stderr is not None else sys.stderr
    external = env.get("TRANS_SERVER_URL", "").strip()
    if external:
        base = external.rstrip("/")
        return base, find_router_model(base, debug=debug, stderr=out)
    router_url = chat_url(args.router_host, args.router_port)
    model = find_router_model(router_url, debug=debug, stderr=out)
    if model is not None:
        return router_url, model
    auto_start = env.get("TRANS_AUTO_START", "1").strip() != "0"
    url = ensure_server(args.host, args.port, auto_start, debug=debug)
    return url, None


def translate_once(
    text: str,
    server_url: str,
    from_lang: str | None,
    to_lang: str | None,
    speak_flag: bool,
    speak_input_flag: bool = False,
    speaker: int | None = None,
    input_speaker: int | None = None,
    stdout=None,
    debug: bool = False,
    stderr=None,
    model: str | None = None,
) -> None:
    """Detect (unless told), translate, print, and speak one phrase."""
    if not text.strip():
        return
    direction_source = from_lang
    if direction_source is None:
        try:
            direction_source = detect_language(
                server_url, text, debug=debug, stderr=stderr, model=model
            )
            lang_name = LANGUAGES.get(direction_source, direction_source)
            err = stderr if stderr is not None else sys.stderr
            print(f"{direction_source} ({lang_name})", file=err)
        except TranslateError as err:
            if to_lang is not None:
                direction_source = None  # cannot detect; --to still known
                print(f"warning: {err}", file=sys.stderr)
            else:
                raise
    if direction_source is None:
        # Undetectable input and only --to given: treat source as English
        # unless the target is English, in which case assume Russian.
        direction_source = "ru" if to_lang == "en" else "en"
    target = pick_target(direction_source, to_lang)
    translation = translate(
        server_url, text, direction_source, target,
        debug=debug, stderr=stderr, model=model,
    )
    out = stdout if stdout is not None else sys.stdout
    print(f"{target}: {translation}", file=out)
    if speak_input_flag:
        speak(text, direction_source, debug=debug, stderr=stderr,
              speaker=input_speaker)
    if speak_flag:
        speak(translation, target, debug=debug, stderr=stderr,
              speaker=speaker)


def run_repl(
    server_url: str,
    from_lang: str | None,
    to_lang: str | None,
    speak_flag: bool,
    speak_input_flag: bool = False,
    speaker: int | None = None,
    input_speaker: int | None = None,
    stdout=None,
    debug: bool = False,
    model: str | None = None,
) -> None:
    out = stdout if stdout is not None else sys.stdout
    print("type a phrase in any language; q to quit", file=out)
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
            translate_once(
                line, server_url, from_lang, to_lang, speak_flag,
                speak_input_flag, speaker, input_speaker=input_speaker,
                stdout=out, debug=debug, model=model,
            )
        except TranslateError as err:
            print(f"error: {err}", file=sys.stderr)
        except KeyboardInterrupt:
            print("(interrupted)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.stop:
        if args.phrase:
            print(
                "error: --stop-server takes no other arguments",
                file=sys.stderr,
            )
            return 1
        if stop_server():
            print("server stopped")
            return 0
        print("no recorded server to stop", file=sys.stderr)
        return 1
    try:
        server_url, model_id = resolve_backend(
            args, dict(os.environ), debug=args.debug
        )
    except TranslateError:
        return 1
    try:
        if args.phrase:
            translate_once(
                " ".join(args.phrase),
                server_url,
                args.from_lang,
                args.to_lang,
                args.speak_flag,
                args.speak_input_flag,
                args.speaker,
                input_speaker=args.input_speaker,
                debug=args.debug,
                model=model_id,
            )
        else:
            run_repl(server_url, args.from_lang, args.to_lang,
                     args.speak_flag, args.speak_input_flag, args.speaker,
                     args.input_speaker, debug=args.debug, model=model_id)
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
