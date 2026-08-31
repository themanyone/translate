import io
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trans  # noqa: E402
from trans import (  # noqa: E402
    TranslateError,
    build_prompt,
    detect_direction,
    ensure_server,
    find_player,
    health_url,
    main,
    parse_args,
    resolve_server_url,
    run_repl,
    server_command,
    speak,
    stop_server,
    translate,
    translate_once,
)


class TestDetectDirection(unittest.TestCase):
    def test_ascii_input_is_english(self):
        self.assertEqual(detect_direction("Hello! My name is Eek."), "en")

    def test_cyrillic_input_is_russian(self):
        self.assertEqual(
            detect_direction("Здравствуйте. Меня зовут Иик."), "ru"
        )

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
            {
                "choices": [
                    {"message": {"role": "assistant", "content": content}}
                ]
            }
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
            up.side_effect = [
                URLError("conn refused"), self._ok_response("Ок.")
            ]
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
            cmd[cmd.index("--chat-template-file") + 1],
            str(trans.TEMPLATE_PATH),
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
                    Path, "read_text",
                    return_value="1234\nllama-server -m x"), \
                mock.patch.object(Path, "exists", return_value=True):
            self.assertTrue(stop_server(Path("/tmp/fake-state")))
        # first kill call: SIGTERM to recorded pid; second (probe) raised
        self.assertEqual(kill_calls[0][0], 1234)

    def test_stop_server_missing_pidfile_returns_false(self):
        with mock.patch.object(Path, "exists", return_value=False):
            self.assertFalse(stop_server(Path("/tmp/fake-state")))

    def test_stop_server_unlinks_stale_pidfile(self):
        # recorded pid no longer exists -> stop returns False but cleans up
        with tempfile.TemporaryDirectory() as state:
            state_dir = Path(state)
            pid_file = state_dir / "server.pid"
            pid_file.write_text("999999999\nllama-server -m x\n")
            with mock.patch("trans.os.kill", side_effect=ProcessLookupError):
                self.assertFalse(stop_server(state_dir))
            self.assertFalse(pid_file.exists())

    def test_stop_server_unlinks_pidfile_after_kill(self):
        kill_calls: list = []

        def record_kill(pid, sig):
            kill_calls.append((pid, sig))
            if sig == signal.SIGTERM:
                return None
            raise ProcessLookupError()  # probe after SIGTERM says it's gone

        with tempfile.TemporaryDirectory() as state:
            state_dir = Path(state)
            pid_file = state_dir / "server.pid"
            pid_file.write_text("1234\nllama-server -m x\n")
            with mock.patch("trans.os.kill", side_effect=record_kill), \
                    mock.patch("trans.time.sleep"):
                self.assertTrue(stop_server(state_dir))
            self.assertFalse(pid_file.exists())
        self.assertEqual(kill_calls[0], (1234, signal.SIGTERM))

    def test_ensure_server_no_spawn_when_lock_held(self):
        # another instance holds the start lock: this one must not spawn a
        # second llama-server; it waits for the existing one to be healthy
        with tempfile.TemporaryDirectory() as state:
            with mock.patch("trans.STATE_DIR", Path(state)):
                lock_fd = trans._acquire_start_lock(Path(state))
                self.assertIsNotNone(lock_fd)
                try:
                    with mock.patch(
                        "trans.urllib.request.urlopen"
                    ) as up, mock.patch(
                        "trans.subprocess.Popen"
                    ) as popen:
                        # health check fails twice, then server appears
                        up.side_effect = [
                            URLError("refused"),   # pre-lock health check
                            URLError("refused"),   # under-lock re-check
                            self._health_ok(),     # wait_for_server poll
                        ]
                        url = ensure_server(self.HOST, self.PORT,
                                            auto_start=True)
                    self.assertEqual(url, "http://127.0.0.1:8144")
                    popen.assert_not_called()
                finally:
                    os.close(lock_fd)


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
                mock.patch("trans.subprocess.Popen") as popen, \
                mock.patch("trans.subprocess.run") as run:
            run.return_value.returncode = 0
            popen.side_effect = self._fake_procs()
            self.assertTrue(speak("Привет!", "en"))
        piper_argv = popen.call_args_list[0][0][0]
        self.assertEqual(piper_argv[0], "/home/k/.local/sbin/piper")
        self.assertIn("--cuda", piper_argv)
        self.assertEqual(
            piper_argv[piper_argv.index("--model") + 1],
            "/home/k/.cache/piper/ru_RU-irina-medium.onnx",
        )
        wav_arg = piper_argv[piper_argv.index("-f") + 1]
        self.assertTrue(wav_arg.endswith(".wav"))
        run_argv = run.call_args[0][0]
        self.assertEqual(run_argv[0], "pw-play")
        self.assertEqual(run_argv[1], wav_arg)

    def test_speak_english_voice_for_russian_direction(self):
        with mock.patch("trans.find_player", return_value="pw-play"), \
                mock.patch("trans.subprocess.Popen") as popen, \
                mock.patch("trans.subprocess.run") as run:
            run.return_value.returncode = 0
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
                mock.patch("trans.subprocess.run"), \
                mock.patch("sys.stderr"):
            popen.side_effect = [bad_piper, player]
            self.assertFalse(speak("hi", "en"))

    def test_speak_feeds_text_to_piper_stdin(self):
        with mock.patch("trans.find_player", return_value="pw-play"), \
                mock.patch("trans.subprocess.Popen") as popen, \
                mock.patch("trans.subprocess.run") as run:
            run.return_value.returncode = 0
            piper, player = self._fake_procs()
            popen.side_effect = [piper, player]
            self.assertTrue(speak("text here", "en"))
        piper.stdin.write.assert_called_once_with(b"text here")
        piper.stdin.close.assert_called_once()

    def test_speak_cleans_up_temp_wav(self):
        created: list = []

        class FakeTemp:
            name = "/tmp/fake-trans.wav"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch("trans.find_player", return_value="pw-play"), \
                mock.patch("trans.subprocess.Popen") as popen, \
                mock.patch("trans.subprocess.run") as run, \
                mock.patch("trans.tempfile.NamedTemporaryFile",
                           side_effect=lambda **kw:
                           (created.append(kw), FakeTemp())[1]), \
                mock.patch("trans.os.unlink") as unlink:
            run.return_value.returncode = 0
            popen.side_effect = self._fake_procs()
            self.assertTrue(speak("clean me", "en"))
        unlink.assert_called_once_with("/tmp/fake-trans.wav")


class TestCLI(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = parse_args(["hello", "world"])
        self.assertEqual(args.phrase, ["hello", "world"])
        self.assertTrue(args.speak_flag)
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8144)
        self.assertFalse(args.stop)

    def test_parse_args_no_speak(self):
        self.assertFalse(parse_args(["--no-speak", "hi"]).speak_flag)

    def test_parse_args_port(self):
        self.assertEqual(parse_args(["--port", "9000", "hi"]).port, 9000)

    def test_parse_args_stop_server(self):
        self.assertTrue(parse_args(["--stop-server"]).stop)

    def test_parse_args_empty_phrase_is_repl(self):
        self.assertEqual(parse_args([]).phrase, [])

    def test_parse_args_rejects_unknown_flag(self):
        with self.assertRaises(SystemExit), \
                mock.patch("sys.stderr"):
            parse_args(["--bogus", "hi"])

    def test_resolve_url_env_override_disables_autostart(self):
        external, host_port, auto = resolve_server_url(
            parse_args(["--port", "1234", "hi"]),
            {"TRANS_SERVER_URL": "http://elsewhere:9999"},
        )
        self.assertEqual(external, "http://elsewhere:9999")
        self.assertIsNone(host_port)
        self.assertFalse(auto)

    def test_resolve_url_defaults(self):
        external, host_port, auto = resolve_server_url(parse_args(["hi"]), {})
        self.assertIsNone(external)
        self.assertEqual(host_port, ("127.0.0.1", 8144))
        self.assertTrue(auto)

    def test_resolve_url_autostart_env_off(self):
        external, host_port, auto = resolve_server_url(
            parse_args(["hi"]), {"TRANS_AUTO_START": "0"}
        )
        self.assertFalse(auto)

    def test_translate_once_prints_output_language_prefix(self):
        out = io.StringIO()
        with mock.patch("trans.translate", return_value="Привет!"), \
                mock.patch("trans.speak", return_value=True) as sp:
            translate_once("Hello!", "http://x", True, stdout=out)
        self.assertEqual(out.getvalue(), "ru: Привет!\n")
        sp.assert_called_once_with("Привет!", "en")

    def test_translate_once_no_speak(self):
        out = io.StringIO()
        with mock.patch("trans.translate", return_value="Hello."), \
                mock.patch("trans.speak") as sp:
            translate_once("Привет.", "http://x", False, stdout=out)
        self.assertEqual(out.getvalue(), "en: Hello.\n")
        sp.assert_not_called()

    def test_translate_once_blank_input_noop(self):
        out = io.StringIO()
        with mock.patch("trans.translate") as tr:
            translate_once("   ", "http://x", True, stdout=out)
        tr.assert_not_called()
        self.assertEqual(out.getvalue(), "")

    def test_run_repl_translates_lines_until_quit(self):
        out = io.StringIO()
        with mock.patch("trans.translate", return_value="Привет!"), \
                mock.patch("trans.speak", return_value=True), \
                mock.patch("builtins.input",
                           side_effect=["Hello!", "   ", "q"]):
            run_repl("http://x", True, stdout=out)
        self.assertEqual(out.getvalue().count("ru: Привет!\n"), 1)

    def test_run_repl_survives_translate_error(self):
        out = io.StringIO()
        with mock.patch("trans.translate",
                        side_effect=TranslateError("boom")), \
                mock.patch("trans.speak"), \
                mock.patch("builtins.input",
                           side_effect=["Hello!", "q"]), \
                mock.patch("sys.stderr"):
            run_repl("http://x", True, stdout=out)

    def test_main_stop_server(self):
        with mock.patch("trans.stop_server", return_value=True) as stop, \
                mock.patch("sys.stdout"):
            code = main(["--stop-server"])
        self.assertEqual(code, 0)
        stop.assert_called_once()

    def test_main_stop_server_rejects_extra_args(self):
        with mock.patch("trans.stop_server") as stop, \
                mock.patch("sys.stderr"):
            self.assertEqual(main(["--stop-server", "hi"]), 1)
        stop.assert_not_called()

    def test_main_one_shot(self):
        with mock.patch("trans.ensure_server", return_value="http://x"), \
                mock.patch("trans.translate", return_value="Привет!"), \
                mock.patch("trans.speak", return_value=True), \
                mock.patch("sys.stdout"):
            code = main(["Hello!"])
        self.assertEqual(code, 0)

    def test_main_one_shot_error_exits_nonzero(self):
        with mock.patch("trans.ensure_server", return_value="http://x"), \
                mock.patch(
                    "trans.translate",
                    side_effect=TranslateError("bad")), \
                mock.patch("sys.stderr"), \
                mock.patch("sys.stdout"):
            self.assertEqual(main(["Hello!"]), 1)

    def test_main_fatal_error_returns_one(self):
        with mock.patch("trans.ensure_server",
                        side_effect=TranslateError("no server")), \
                mock.patch("sys.stderr"):
            self.assertEqual(main(["Hello!"]), 1)

    def test_main_uses_external_url_and_skips_ensure_server(self):
        # mocks only the network seam: proves the TRANS_SERVER_URL plumbing
        # actually reaches the HTTP layer (regression test for the
        # URL-discard bug).
        ok_body = json.dumps(
            {"choices": [
                {"message": {"role": "assistant", "content": "Привет!"}}
            ]}
        ).encode()
        with mock.patch.dict(os.environ,
                             {"TRANS_SERVER_URL": "http://elsewhere:9999"}), \
                mock.patch("trans.ensure_server") as ensure, \
                mock.patch("trans.urllib.request.urlopen") as up, \
                mock.patch("trans.speak", return_value=True):
            resp = mock.MagicMock()
            resp.read.return_value = ok_body
            resp.__enter__.return_value = resp
            up.return_value = resp
            self.assertEqual(main(["Hello!"]), 0)
        ensure.assert_not_called()
        self.assertEqual(up.call_args[0][0].full_url,
                         "http://elsewhere:9999/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
