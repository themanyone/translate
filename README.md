# trans

English <-> Russian translator that prints the translation and speaks it
with piper TTS. Language direction is detected automatically from the script
of what you type (Cyrillic -> English, otherwise -> Russian).

## Requirements

- llama.cpp's `llama-server` (on PATH)
- TranslateGemma 4B GGUF (already at the hardcoded model path)
- piper TTS at `~/.local/sbin/piper` with the `ru_RU-irina-medium` and
  `en_US-libritts_r-medium` voices
- a WAV player: `pw-play`, `paplay`, or `aplay`

## Install

```sh
ln -sf "$PWD/trans.py" ~/.local/bin/trans
```

## Usage

- `trans` - interactive mode; type phrases, `q` quits.
- `trans <phrase...>` - translate one phrase and exit.
- `trans --no-speak <phrase...>` - print only.
- `trans --stop-server` - stop the auto-started llama-server.
- `trans --version` / `trans --help` - version and usage.

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
