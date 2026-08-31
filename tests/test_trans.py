import unittest

from trans import build_prompt, detect_direction


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


if __name__ == "__main__":
    unittest.main()
