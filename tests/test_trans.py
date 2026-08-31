import json
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

from trans import TranslateError, build_prompt, detect_direction, translate


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
