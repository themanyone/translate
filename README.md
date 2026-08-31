# trans

Translator that prints the translation and speaks it aloud with piper TTS
(piper plays the audio itself, no player needed). The input language is
detected automatically by the running translation model; the output
language defaults to English, except English input defaults to Spanish.

## Requirements

- llama.cpp's `llama-server` (on PATH)
- TranslateGemma 4B GGUF (already at the hardcoded model path)
- piper TTS at `~/.local/sbin/piper`
- `download_voices` at `~/.local/sbin/download_voices` (used to fetch a
  voice on demand when the output language has none cached)

## Install

```sh
ln -sf "$PWD/trans.py" ~/.local/bin/trans
```

## Usage

- `trans` - interactive mode; type phrases, `q` quits.
- `trans <phrase...>` - translate one phrase and exit.
- `trans --from <lang> <phrase...>` - skip detection; e.g. `--from
  Russian` or `--from ru`.
- `trans --to <lang> <phrase...>` - choose the output language; e.g.
  `--to Spanish`.
- `trans --no-speak <phrase...>` - print only.
- `trans --stop-server` - stop the auto-started llama-server.
- `trans --version` / `trans --help` - version and usage.

Without `--to`: any detected input language translates to English; English
input translates to Spanish. Languages are given by name (`French`) or
code (`fr`); unrecognized names are rejected.

Exit codes: 0 on success, 1 on errors, 130 when interrupted with Ctrl+C.
In interactive mode you can quit with `q`, `quit`, or `exit`, or Ctrl+D.

The app talks to `llama-server` at `127.0.0.1:8144` (override with
`--host`/`--port`). By default it auto-starts a detached server if none is
running and leaves it running afterwards. Server logs live in
`~/.local/state/trans/server.log`.

Set `TRANS_AUTO_START=0` to manage the server yourself: the app then prints
the exact command it wants you to run and exits if the server is missing.
Set `TRANS_SERVER_URL=http://host:port` to use an external server entirely.

## Development

```sh
python3 tests/test_trans.py -v
```
