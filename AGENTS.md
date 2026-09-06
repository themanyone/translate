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

## Testing

All 87 tests pass, including the new test for `--speak-input` functionality.

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