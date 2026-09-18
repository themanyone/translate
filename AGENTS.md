# translate — Speech Translator Status

## Current State

This project implements a command-line translation app with speech-to-text capabilities. **Added the `--speak-input` feature** to speak the input text in its original language before translating.

## Key Changes

### 1. New CLI Option: `--speak-input`
- Speaks the input text in its original language (detected or specified via `--from`)
- Works independently alongside existing `--speak` and `--no-speak` flags
- Default: Input speaking is disabled (`False`)

### 2. Updated Functions
- `translate_once()`: Added `speak_input_flag` parameter
- `run_repl()`: Added `speak_input_flag` parameter  
- `parse_args()`: Added `--speak-input` argument
- `main()`: Pass `speak_input_flag` to both `translate_once()` and `run_repl()`

### 3. Added Tests
- `test_translate_once_speaks_input()`: Verifies both input and output speech occur when `--speak-input` is enabled

### 4. Updated Documentation
- `README.md`: Added `--speak-input` usage entry
- Existing design docs remain accurate (feature generalizes input language detection for speaking)

### 5. Speaker Selection: `-s` / `--speaker`
- `parse_args()`: Added `-s N` / `--speaker N` (int, default None)
- `speak()`: Appends `--speaker N` to the piper command when set
- `translate_once()` and `run_repl()`: Thread the `speaker` value through to `speak()`

### 6. Input Speaker: `-si` / `--speak-input [N]`
- `_extract_speak_input()`: Pulls `-si`/`--speak-input` out of argv, accepting
  an optional speaker id attached (`-si2`, `-si=2`, `--speak-input=2`) or
  separate (`-si 2`); a non-numeric token stays a phrase, so the bare flag
  still works before a phrase
- `parse_args()`: Exposes the value as `args.input_speaker`
- `translate_once()` and `run_repl()`: Input speech uses `input_speaker`;
  output speech keeps using `speaker`

### 7. Multi-Speaker Voice Selection: `-s` / `-si`
- `MULTI_SPEAKER_VOICES`: known multi-speaker piper voices per language,
  taken from the official catalog (`voices.json`); most voices have only
  one speaker, so a speaker id would otherwise be a silent no-op
- `_voice_num_speakers()`: reads `num_speakers` from a voice's `.onnx.json`
- `voice_for_language(..., multi_speaker=True)`: prefers a multi-speaker
  voice already on disk, downloads the known one if needed, and otherwise
  falls back to the normal voice
- `speak()`: when the resolved voice has one speaker, warns that the
  speaker id has no effect instead of passing a meaningless `--speaker`

## Testing

All 101 tests pass, covering `--speak-input`, `-s`, `-si`, and multi-speaker
voice selection.

## Behavior

### Without any speak flags:
- Translation only
- Output spoken (default)
- Input not spoken

### With `--no-speak`:
- Translation only
- No output speaking
- Input not spoken

### With `--speak-input`:
- Translation and input speaking
- Output speaking (default)
- Input spoken in original language

### With `--speak-input` + `--no-speak`:
- Translation only
- No output speaking
- Input spoken in original language

## Impact

This feature provides users with the ability to:
1. Hear the original input text before seeing its translation
2. Work with text in unfamiliar languages by hearing it first
3. Maintain existing behavior by not using `--speak-input`
4. Fully customize speaking behavior with combinations of flags