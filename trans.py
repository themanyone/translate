#!/usr/bin/env python3
"""trans - English<->Russian translator that prints and speaks the result.

Direction is auto-detected (Cyrillic script check). Translation runs on a
local llama-server hosting TranslateGemma 4B; speech uses piper TTS.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
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
    "/home/k/.cache/huggingface/hub/"
    "models--mradermacher--translategemma-4b-it-i1-GGUF/"
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


def _acquire_start_lock(state_dir: Path) -> int | None:
    """Hold an exclusive lock for the check-then-spawn window.

    Returns the open lock fd, or None if another trans instance is
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
            _start_server_detached(host, port, STATE_DIR)
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
    """Say the translation aloud; False on any failure (never raises).

    piper writes a WAV to a temp file and the player plays the file:
    none of pw-play/paplay/aplay on this system accept WAV on stdin,
    and pw-play rejects "-" and /dev/stdin outright.
    """
    voice = VOICES["ru" if direction == "en" else "en"]
    player = find_player()
    if player is None:
        print(
            "warning: no audio player found (pw-play, paplay, aplay)",
            file=sys.stderr,
        )
        return False
    wav_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as wav_file:
            wav_path = wav_file.name
        piper_proc = subprocess.Popen(
            [PIPER, "--cuda", "--model", voice, "-f", wav_path],
            stdin=subprocess.PIPE,
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
        player_rc = subprocess.run(
            [player, wav_path], check=False
        ).returncode
        if player_rc != 0:
            print(
                f"warning: speech pipeline failed (player={player_rc})",
                file=sys.stderr,
            )
            return False
        return True
    except OSError as err:
        print(f"warning: speech failed: {err}", file=sys.stderr)
        return False
    finally:
        if wav_path is not None:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


__version__ = "1.0.0"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trans",
        description="Translate between English and Russian, "
        "print and speak it.",
    )
    parser.add_argument(
        "phrase", nargs="*",
        help="phrase to translate (omit for interactive mode)",
    )
    parser.add_argument(
        "--no-speak", dest="speak_flag", action="store_false",
        help="print the translation without speaking it",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"server host (default {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"server port (default {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--stop-server", dest="stop", action="store_true",
        help="stop the auto-started llama-server and exit",
    )
    parser.add_argument(
        "--version", action="version", version=f"trans {__version__}"
    )
    return parser.parse_args(argv)


def resolve_server_url(
    args: argparse.Namespace, env: dict[str, str]
) -> tuple[str, tuple[str, int] | None, bool]:
    """Return (external_url, host_port, auto_start).

    external_url is non-None only when TRANS_SERVER_URL points at a server
    whose lifecycle we must never manage; host_port is None in that case.
    """
    external = env.get("TRANS_SERVER_URL", "").strip()
    if external:
        return external.rstrip("/"), None, False
    auto_start = env.get("TRANS_AUTO_START", "1").strip() != "0"
    return None, (args.host, args.port), auto_start


def translate_once(
    text: str, server_url: str, speak_flag: bool, stdout=None
) -> None:
    direction = detect_direction(text)
    if direction is None:
        return
    translation = translate(server_url, text, direction)
    out = stdout if stdout is not None else sys.stdout
    print(f"{LANG_NAMES[direction][1][:2].lower()}: {translation}", file=out)
    if speak_flag:
        speak(translation, direction)


def run_repl(server_url: str, speak_flag: bool, stdout=None) -> None:
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
    external_url, host_port, auto_start = resolve_server_url(
        args, dict(os.environ)
    )
    try:
        if external_url is not None:
            server_url = external_url
        elif host_port is not None:
            host, port = host_port
            server_url = ensure_server(host, port, auto_start)
        else:  # pragma: no cover - resolve_server_url always fills one
            raise TranslateError("no server configured")
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
