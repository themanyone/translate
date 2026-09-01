# trans

Translator that prints the translation and speaks it aloud with piper TTS
(piper plays the audio itself, no player needed). The input language is
detected automatically by the running translation model; the output
language defaults to English, except English input defaults to Spanish.

## Requirements

- llama.cpp's `llama-server` (on PATH), run either way:
  - **router mode** (preferred): a server already running that hosts the
    TranslateGemma model among its models, e.g.
    `llama-server --jinja --models-dir /win/models --models-preset
    /win/models/.models.ini --host 0.0.0.0 --port 8087 --no-warmup
    --models-max 1 --sleep-idle-seconds 420 -t 6 -fa on -np 1 --kv-unified
    --no-mmproj-offload`
  - **dedicated mode** (fallback): trans auto-starts a single-model
    server with the TranslateGemma GGUF at the hardcoded model path
- piper TTS at `/home/k/.local/sbin/piper`
- `download_voices` at `/home/k/.local/sbin/download_voices` (used to fetch a
  voice on demand when the output language has none cached)

## Install

```sh
ln -sf "$PWD/trans.py" /home/k/.local/bin/trans
```

## Usage

- `trans` - interactive mode; type phrases, `q` quits.
- `trans <phrase...>` - translate one phrase and exit.
- `trans --from <lang> <phrase...>` - skip detection; e.g. `--from
  Russian` or `--from ru`.
- `trans --to <lang> <phrase...>` - choose the output language; e.g.
  `--to Spanish`.
- `trans --no-speak <phrase...>` - print only.
- `trans --router-port <port>` - router server port (default 8087);
  `--router-host` likewise (default 127.0.0.1)
- `trans --stop-server` - stop the auto-started dedicated llama-server.
- `trans --version` / `trans --help` - version and usage.

Without `--to`: any detected input language translates to English; English
input translates to Spanish. Languages are given by name (`French`) or
code (`fr`); unrecognized names are rejected.

Exit codes: 0 on success, 1 on errors, 130 when interrupted with Ctrl+C.
In interactive mode you can quit with `q`, `quit`, or `exit`, or Ctrl+D.

## How the translation server is found

trans prefers a router-mode llama-server that is already running on
`127.0.0.1:8087` (override with `--router-host`/`--router-port`): it
queries `/v1/models` for a TranslateGemma model (id containing
`translategemma`, or alias `trans`) and then names that model in every
chat request. If the router is not running or hosts no such model,
trans falls back to the dedicated single-model server at
`127.0.0.1:8144` (override with `--host`/`--port`), auto-starting a
detached `llama-server` if none is running and leaving it running
afterwards. `--stop-server` stops that dedicated server only; the
router is started/stopped externally and trans never manages it.
Server logs for the dedicated server live in
`/home/k/.local/state/trans/server.log`.

Set `TRANS_AUTO_START=0` to manage the dedicated server yourself: when
no router is found, the app prints the exact command it wants you to
run and exits if the server is missing.
Set `TRANS_SERVER_URL=http://host:port` to use one fixed server entirely
(its lifecycle is never managed); model selection still applies when
that server hosts TranslateGemma.

## Development

```sh
python3 tests/test_trans.py -v
```
