# trans — Speech Translator: Design

Status: v1.0.0 implements the original English↔Russian pair; this revision
generalizes input-language detection and output-language selection to
arbitrary languages (not yet implemented).

## Purpose

A command-line app: give it a phrase in unknown tongue, it translates to [Language] (or vice versa), prints the translation, and speaks it aloud with TTS. Input language is detected automatically. Output language defaults to English.

## Ingredients (all pre-existing on this machine)

- **TranslateGemma 4B** GGUF (mradermacher i1-IQ4_NL quant) served by `llama-server` from `llama.cpp` 0.3.0-dev.
- **Chat template** `translategemma.jinja` (in repo). It expects structured user content (`source_lang_code`/`target_lang_code`); when served plainly with `--no-jinja --chat-template-file`, a plain-string user message passes through and the model follows instruction text embedded in the prompt.
- **Piper TTS** at `/home/k/.local/sbin/piper` with `--cuda` flag.
- **download_voices** at `/home/k/.local/sbin/download_voices` for listing and fetching piper voices.
- Put a list of configurable dirs at top of code.

## Key findings from experiments (2026-08-30)

1. **Short prompt → chatty output.** `Translate to Russian: Hello! My name is Eek.` yields formal/informal options plus explanations. Unusable for a translator.
2. **Long instruction prompt → clean single translation.** Wrapping the input as
   `Translate the following {src} text into {tgt}. Produce only the {tgt} translation, without any additional explanations or commentary: {text}`
   produces exactly the translation, both directions, at `--temp 0`. The shape is language-agnostic: the instruction itself names the source and target languages, so it works for arbitrary pairs.
3. **llama-cli is awkward to parse** (banner, `[end of text]`, timings on stdout) and pays ~10-15 s model-load per invocation. **llama-server REST** (`/v1/chat/completions`) returns clean JSON with just the translation; ~2-3 s per phrase after a one-time load (~4 GB into an 8 GB RTX 3070 Laptop; default 4 slots, 131072 ctx each).

## Architecture

Single-file stdlib Python app (no pip dependencies): `translate.py` installed to `/home/k/.local/sbin/translate.py`, source lives in this repo.

### Components

1. **Server manager**
   - Checks `GET {url}/health`; expects `{"status":"ok"}`.
   - If unreachable and auto-start enabled: launches `llama-server` detached (nohup-style, own process group), state under `/home/k/.local/state/trans/` (`server.log`, `server.pid`), polls health up to ~120 s.
   - If unreachable and auto-start disabled: prints the exact launch command and exits non-zero.
   - Never stops the server on app exit; `trans --stop-server` is the explicit off-switch (reads `server.pid`, SIGTERM, waits).
   - Address configurable: `--port` (default 8144), `--host` (default 127.0.0.1), env `TRANS_SERVER_URL` (full URL override), `TRANS_AUTO_START` (default `1`; `0` disables).

2. **Language detection + direction**
   - Detect arbitrary input language, unless `--from [language]` is specified. `--from` takes a language name or code matched like `English` → `en_US`, `Russian` → `ru_RU`, etc.
   - Without `--from`, use a detection prompt against the running llama-server to detect the language code of the input text.
   - `--to [language]` optionally sets the output language. If no `--to`, assume English output. Or, if English input is given (detected), choose Spanish output instead.

3. **Translation client**
   - POST `/v1/chat/completions`, single user message = long instruction prompt (finding #2) with the resolved source/target language names filled in, `temperature` pinned via server-side `--temp 0` (client sends no sampling params).
   - Timeout ~180 s. Retries once on transient connection errors.

4. **Speaker**
   - `piper --cuda --model ~/.cache/piper/<voice> <<< "text"` Output is played automatically.
   - Voice chosen by matching language code in `~/.cache/piper/` e.g.: Russian → `ru_RU-irina-medium`, English → `en_US-libritts_r-medium`.
   - If no matching `{language}*` file in `~/.cache/piper/`, use `download_voices` to obtain a list of available voices. Pick the first one matching `{language}*`, e.g. `fr_FR` => `fr_FR-mls-medium`
   - Download chosen voice via `download_voices --download-dir ~/.cache/piper/ [voice...]`
   - Long text is passed as-is; piper handles multi-sentence input natively.
   - Speech failures print a warning; never abort the translation loop.

### CLI surface

- `trans` → interactive REPL. Each line: detect, translate, print `<langcode>: <text>` (prefix is the output language code), speak. `q` or `exit` (or EOF) quits. `Ctrl+C` interrupts cleanly.
- `trans <phrase>...` → one-shot: translate, print, speak, exit.
- Flags: `--from [lang]`, `--to [lang]`, `--no-speak`, `--host`, `--port`, `--stop-server`, `--help`, `--version`.
- Unknown flags rejected with usage.

### Error handling

- Server not reachable after auto-start attempt (timeout) → clear message pointing at `/home/k/.local/state/trans/server.log`.
- HTTP error / malformed JSON / empty content → report and continue (REPL) or exit non-zero (one-shot).
- Unrecognized `--from`/`--to` language, or undetectable input language → report and continue (REPL) or exit non-zero (one-shot).
- Piper/play/voice-download failure → warn, continue.
- KeyboardInterrupt mid-translation → back to prompt (REPL) or clean exit.

### Testing

- Manual end-to-end against a live server: en→ru and ru→en (the original pair), plus at least one other language (e.g. detected `fr` input, or `--to French`) to exercise LLM-based detection, `--from`/`--to` overrides, and the voice listing/download fallback.
- Piper audio heard; `--no-speak` path; auto-start and manual modes; `--stop-server`.

## Non-goals

- No microphone/speech-input (text in only).
- No offline language-detection library (the running model does detection).
- No translation memory or history, no GUI.
