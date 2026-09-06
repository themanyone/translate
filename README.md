# translate

Text to speech (TTS) with automatic language detection and translation.
Defaults: Anything-to-English & English-to-Spanish. Configurable.

For English-to-English, just use [piper TTS](https://github.com/OHF-Voice/piper1-gpl) directly.

## Requirements

- A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server router, or `llama-server` available in path.
  - **router mode** (preferred): a server already running that hosts the
    TranslateGemma model among its models. Configure location & port, e.g.
    `llama-server --jinja --models-dir /my/models --models-preset
    /path/to/.models.ini --host 0.0.0.0 --port 8087 --no-warmup
    --models-max 1 --sleep-idle-seconds 420 -t 6 -fa on -np 1 --kv-unified
    --no-mmproj-offload` (see `models.ini` below).
  - **dedicated mode** (failsafe fallback): translate auto-starts `llama-server`
    with the TranslateGemma GGUF at your hardcoded model path
- [piper TTS](https://github.com/OHF-Voice/piper1-gpl) installed (linked) somewhere in path.
- `download_voices` from piper TTS, also linked to path (used to fetch a
  voice on demand when the output language has none cached)

## Install

```sh
ln -sf "$PWD/translate.py" ~/.local/bin/translate
```

## Install dependencies

Install [piper TTS](https://github.com/OHF-Voice/piper1-gpl) following updated
instructions. To get development version, we did this:

```sh
git clone https://github.com/OHF-voice/piper1-gpl.git
cd piper1-gpl
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .[dev]
script/dev_build
python3 -m build
```

Create links to make piper accessable anywhere.

```sh
ln -srf .venv/bin/piper ~/.local/bin/
ln -srf ./script/download_voices ~/.local/bin/
```

Set up your router server `models.ini` to include TranslateGemma.

```sh
[mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL]
alias = trans
no-jinja = true
chat-template-file = /path/to/translate/translategemma.jinja
```

Use our `translategemma.jinja` template because builtin one has never worked.

Launch from command line one time to obtain model & chat with it.
`llama-cli -hf mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL --chat-template-file translategemma.jinja`

Launch server in router mode by specifying `--models-dir`

```sh
llama-server --jinja --models-dir /my/models --models-preset
    /path/to/.models.ini --host 0.0.0.0 --port 8087 --no-warmup
    --models-max 1 --sleep-idle-seconds 420 -t 6 -fa on -np 1 --kv-unified
    --no-mmproj-offload
```

## Usage

- `translate` - interactive mode; type phrases, `q` quits.
- `translate <phrase...>` - translate one phrase and exit.
- `translate --from <lang> <phrase...>` - skip detection; e.g. `--from
  Russian` or `--from ru`.
- `translate --to <lang> <phrase...>` - choose the output language; e.g.
  `--to Spanish`.
- `translate --no-speak <phrase...>` - print only.
- `translate --router-port <port>` - router server port (default 8087);
  `--router-host` likewise (default 127.0.0.1)
- `translate --stop-server` - stop the auto-started dedicated llama-server.
- `translate --version` / `translate --help` - version and usage.

Without `--to`: any detected input language translates to English; English
input translates to Spanish. Languages are given by name (`French`) or
code (`fr`); unrecognized language names are rejected.

Exit codes: 0 on success, 1 on errors, 130 when interrupted with Ctrl+C.
In interactive mode you can quit with `q`, `quit`, or `exit`, or Ctrl+D.

## How the translation server is found

`translate` looks for a router-mode llama-server that is already running on
`127.0.0.1:8087` (override with `--router-host`/`--router-port`): it
queries `/v1/models` for a TranslateGemma model (id containing
`translategemma`, or alias `translate`) and then names that model in every
chat request. If the router is not running or hosts no such model,
translate falls back to launchinkg a dedicated single-model server at
`127.0.0.1:8144` (override with `--host`/`--port`), auto-starting a
detached `llama-server` if none is running and leaving it running
afterwards. `--stop-server` stops that dedicated server only.

Your router is started/stopped externally and translate never manages it.
Server logs for the dedicated server live in
`~/.local/state/translate/server.log`.

Set `TRANS_AUTO_START=0` to manage the dedicated server yourself: when
no router is found, the app prints the exact command it wants you to
run and exits if the server is missing.
Set `TRANS_SERVER_URL=http://host:port` to use one fixed server entirely
(its lifecycle is never managed); model selection still applies when
that server hosts TranslateGemma.

## Testing

```sh
python3 tests/test_translate.py -v
```

### Thanks for trying out themanyone/translate

- GitHub https://github.com/themanyone
- YouTube https://www.youtube.com/themanyone
- Mastodon https://mastodon.social/@themanyone
- Linkedin https://www.linkedin.com/in/henry-kroll-iii-93860426/
- Buy me a coffee https://buymeacoffee.com/isreality
- [TheNerdShow.com](http://thenerdshow.com/)

Copyright (C) 2026 Henry Kroll III, www.thenerdshow.com.
See [LICENSE](LICENSE) for details.
