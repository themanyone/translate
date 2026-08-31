import io
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

import trans
from trans import (
    TranslateError,
    build_prompt,
    detect_direction,
    ensure_server,
    find_player,
    health_url,
    server_command,
    speak,
    stop_server,
    translate,
)


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


class TestServerManager(unittest.TestCase):
    HOST, PORT = "127.0.0.1", 8144

    def _health_ok(self, body=b'{"status":"ok"}') -> mock.MagicMock:
        resp = mock.MagicMock()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        return resp

    def test_urls(self):
        self.assertEqual(
            health_url("127.0.0.1", 8144), "http://127.0.0.1:8144/health"
        )

    def test_server_command_is_verbatim(self):
        cmd = server_command("127.0.0.1", 8144)
        self.assertEqual(cmd[0], "llama-server")
        self.assertIn("--no-jinja", cmd)
        self.assertIn("--temp", cmd)
        self.assertEqual(cmd[cmd.index("--temp") + 1], "0")
        self.assertEqual(
            cmd[cmd.index("--chat-template-file") + 1], str(trans.TEMPLATE_PATH)
        )
        self.assertEqual(cmd[cmd.index("-m") + 1], trans.MODEL_PATH)
        self.assertIn("--port", cmd)

    def test_ensure_server_returns_url_when_healthy(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._health_ok()
            url = ensure_server(self.HOST, self.PORT, auto_start=True)
        self.assertEqual(url, "http://127.0.0.1:8144")

    def test_ensure_server_starts_detached_when_unhealthy(self):
        with (
            mock.patch("trans.urllib.request.urlopen") as up,
            mock.patch("trans.subprocess.Popen") as popen,
            mock.patch("trans.wait_for_server", return_value=True) as wait,
            mock.patch("trans.STATE_DIR", Path("/tmp/fake-state")),
        ):
            up.side_effect = URLError("refused")
            popen.return_value.pid = 4242
            url = ensure_server(self.HOST, self.PORT, auto_start=True)
        self.assertEqual(url, "http://127.0.0.1:8144")
        self.assertEqual(popen.call_count, 1)
        argv = popen.call_args[0][0]
        self.assertEqual(argv[0], "llama-server")
        popen_kwargs = popen.call_args[1]
        self.assertTrue(popen_kwargs["start_new_session"])
        wait.assert_called_once_with("http://127.0.0.1:8144/health")
        pidfile = Path("/tmp/fake-state") / "server.pid"
        self.assertEqual(pidfile.read_text().splitlines()[0], "4242")

    def test_ensure_server_raises_when_start_times_out(self):
        with (
            mock.patch("trans.urllib.request.urlopen") as up,
            mock.patch("trans.subprocess.Popen") as popen,
            mock.patch("trans.wait_for_server", return_value=False),
            mock.patch("trans.STATE_DIR", Path("/tmp/fake-state")),
        ):
            up.side_effect = URLError("refused")
            popen.return_value.pid = 99
            with self.assertRaises(TranslateError), \
                    mock.patch("sys.stderr") as err:
                ensure_server(self.HOST, self.PORT, auto_start=True)
        written = "".join(str(c) for c in err.write.call_args_list)
        self.assertIn("server.log", written)

    def test_ensure_server_manual_mode_prints_command_and_raises(self):
        with mock.patch("trans.urllib.request.urlopen") as up, \
                mock.patch("sys.stderr") as err:
            up.side_effect = URLError("refused")
            with self.assertRaises(TranslateError):
                ensure_server(self.HOST, self.PORT, auto_start=False)
        written = "".join(str(c) for c in err.write.call_args_list)
        self.assertIn("llama-server", written)
        self.assertIn("8144", written)

    def test_stop_server_sigterms_pid(self):
        kill_calls: list = []

        def record_kill(pid, sig):
            kill_calls.append((pid, sig))
            if len(kill_calls) == 1:
                return None
            raise ProcessLookupError()

        with mock.patch("trans.os.kill", side_effect=record_kill), \
                mock.patch("trans.time.sleep"), \
                mock.patch.object(
                    Path, "read_text", return_value="1234\nllama-server -m x"), \
                mock.patch.object(Path, "exists", return_value=True):
            self.assertTrue(stop_server(Path("/tmp/fake-state")))
        # first kill call: SIGTERM to recorded pid; second (probe) raised
        self.assertEqual(kill_calls[0][0], 1234)

    def test_stop_server_missing_pidfile_returns_false(self):
        with mock.patch.object(Path, "exists", return_value=False):
            self.assertFalse(stop_server(Path("/tmp/fake-state")))


class TestSpeaker(unittest.TestCase):
    def _fake_procs(self):
        piper = mock.MagicMock(stdout=mock.MagicMock())
        player = mock.MagicMock()
        for p in (piper, player):
            p.wait.return_value = 0
        return [piper, player]

    def test_find_player_first_on_path(self):
        with mock.patch(
            "trans.shutil.which", side_effect=lambda n: f"/usr/bin/{n}"
        ):
            self.assertEqual(find_player(), "/usr/bin/pw-play")

    def test_find_player_none_when_missing(self):
        with mock.patch("trans.shutil.which", return_value=None):
            self.assertIsNone(find_player())

    def test_speak_russian_voice_for_english_direction(self):
        # direction "en" = English input, so the spoken translation is Russian
        with mock.patch("trans.find_player", return_value="pw-play"), \
                mock.patch("trans.subprocess.Popen") as popen:
            popen.side_effect = self._fake_procs()
            self.assertTrue(speak("Привет!", "en"))
        piper_argv = popen.call_args_list[0][0][0]
        self.assertEqual(piper_argv[0], "/home/k/.local/sbin/piper")
        self.assertIn("--cuda", piper_argv)
        self.assertEqual(
            piper_argv[piper_argv.index("--model") + 1],
            "/home/k/.cache/piper/ru_RU-irina-medium.onnx",
        )
        self.assertEqual(piper_argv[piper_argv.index("-f") + 1], "-")

    def test_speak_english_voice_for_russian_direction(self):
        with mock.patch("trans.find_player", return_value="pw-play"), \
                mock.patch("trans.subprocess.Popen") as popen:
            popen.side_effect = self._fake_procs()
            self.assertTrue(speak("Hello!", "ru"))
        piper_argv = popen.call_args_list[0][0][0]
        self.assertEqual(
            piper_argv[piper_argv.index("--model") + 1],
            "/home/k/.cache/piper/en_US-libritts_r-medium.onnx",
        )

    def test_speak_returns_false_without_player(self):
        with mock.patch("trans.find_player", return_value=None):
            self.assertFalse(speak("hi", "en"))

    def test_speak_returns_false_when_piper_fails(self):
        bad_piper, player = self._fake_procs()
        bad_piper.wait.return_value = 1
        with mock.patch("trans.find_player", return_value="pw-play"), \
                mock.patch("trans.subprocess.Popen") as popen, \
                mock.patch("sys.stderr"):
            popen.side_effect = [bad_piper, player]
            self.assertFalse(speak("hi", "en"))

    def test_speak_feeds_text_to_piper_stdin(self):
        with mock.patch("trans.find_player", return_value="pw-play"), \
                mock.patch("trans.subprocess.Popen") as popen:
            piper, player = self._fake_procs()
            popen.side_effect = [piper, player]
            self.assertTrue(speak("text here", "en"))
        piper.stdin.write.assert_called_once_with(b"text here")
        piper.stdin.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
