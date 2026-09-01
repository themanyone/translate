import io
import json
import os
import subprocess
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
    detect_language,
    ensure_server,
    find_router_model,
    health_url,
    main,
    parse_args,
    pick_target,
    resolve_backend,
    resolve_language,
    run_repl,
    server_command,
    speak,
    stop_server,
    translate,
    translate_once,
    voice_for_language,
)


class TestResolveLanguage(unittest.TestCase):
    def test_name_to_code(self):
        self.assertEqual(resolve_language("English"), "en")
        self.assertEqual(resolve_language("Russian"), "ru")
        self.assertEqual(resolve_language("Spanish"), "es")
        self.assertEqual(resolve_language("french"), "fr")

    def test_code_passthrough(self):
        self.assertEqual(resolve_language("en"), "en")
        self.assertEqual(resolve_language("RU"), "ru")

    def test_three_letter_alias(self):
        self.assertEqual(resolve_language("eng"), "en")
        self.assertEqual(resolve_language("rus"), "ru")
        self.assertEqual(resolve_language("fre"), "fr")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            resolve_language("klingon")


class TestPickTarget(unittest.TestCase):
    def test_default_output_is_english(self):
        self.assertEqual(pick_target("fr", None), "en")
        self.assertEqual(pick_target("ru", None), "en")

    def test_english_input_defaults_to_spanish(self):
        self.assertEqual(pick_target("en", None), "es")

    def test_explicit_to_wins(self):
        self.assertEqual(pick_target("en", "ru"), "ru")
        self.assertEqual(pick_target("fr", "es"), "es")


class TestBuildPrompt(unittest.TestCase):
    def test_uses_language_names(self):
        self.assertEqual(
            build_prompt("Hello!", "en", "es"),
            "Translate the following English text into Spanish. "
            "Produce only the Spanish translation, without any additional "
            "explanations or commentary: Hello!",
        )

    def test_russian_to_english(self):
        self.assertEqual(
            build_prompt("Привет!", "ru", "en"),
            "Translate the following Russian text into English. "
            "Produce only the English translation, without any additional "
            "explanations or commentary: Привет!",
        )

    def test_unknown_language_raises(self):
        with self.assertRaises(ValueError):
            build_prompt("hi", "en", "xx")


class TestDetectLanguage(unittest.TestCase):
    SERVER = "http://127.0.0.1:8144"

    def _ok(self, content: str) -> mock.MagicMock:
        resp = mock.MagicMock()
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant",
                                      "content": content}}]}
        ).encode()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        return resp

    def test_returns_detected_code(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._ok("es")
            self.assertEqual(
                detect_language(self.SERVER, "Hola, cómo estás?"), "es"
            )
        sent = json.loads(up.call_args[0][0].data)
        self.assertIn("Hola, cómo estás?", sent["messages"][0]["content"])

    def test_normalizes_three_letter_answer(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._ok("eng")
            self.assertEqual(detect_language(self.SERVER, "Hello"), "en")

    def test_strips_punctuation_from_answer(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._ok('"fr".')
            self.assertEqual(detect_language(self.SERVER, "Bonjour"), "fr")

    def test_unrecognized_answer_raises(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._ok("I am not sure")
            with self.assertRaises(TranslateError):
                detect_language(self.SERVER, "???")

    def test_empty_text_raises(self):
        with self.assertRaises(TranslateError):
            detect_language(self.SERVER, "   ")


class TestTranslate(unittest.TestCase):
    SERVER = "http://127.0.0.1:8144"

    def _ok_response(self, content: str) -> mock.MagicMock:
        resp = mock.MagicMock()
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant",
                                      "content": content}}]}
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
            result = translate(
                self.SERVER, "Hello! My name is Eek.", "en", "ru"
            )
        self.assertEqual(result, "Привет! Меня зовут Ик.")
        body = self._assert_request_body(up)
        self.assertEqual(
            body["messages"],
            [
                {
                    "role": "user",
                    "content": build_prompt(
                        "Hello! My name is Eek.", "en", "ru"
                    ),
                }
            ],
        )
        self.assertNotIn("temperature", body)
        self.assertNotIn("model", body)  # dedicated server: no selector

    def test_model_selector_sent_when_given(self):
        # router-mode server: every request names the model to use
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._ok_response("Hola.")
            translate(
                self.SERVER, "Hello.", "en", "es",
                model="mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL",
            )
        body = self._assert_request_body(up)
        self.assertEqual(
            body["model"],
            "mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL",
        )

    def test_strips_whitespace_from_content(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._ok_response("  Hello.\n")
            result = translate(self.SERVER, "Привет.", "ru", "en")
        self.assertEqual(result, "Hello.")

    def test_retries_once_on_connection_error_then_succeeds(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.side_effect = [
                URLError("conn refused"), self._ok_response("Ок.")
            ]
            result = translate(self.SERVER, "OK.", "en", "ru")
        self.assertEqual(result, "Ок.")
        self.assertEqual(up.call_count, 2)

    def test_raises_after_exhausting_retries(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.side_effect = URLError("conn refused")
            with self.assertRaises(TranslateError):
                translate(self.SERVER, "OK.", "en", "ru")
        self.assertEqual(up.call_count, 2)

    def test_http_error_raises_without_retry(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.side_effect = HTTPError(
                f"{self.SERVER}/v1/chat/completions", 500, "err", {}, None
            )  # type: ignore[arg-type]
            with self.assertRaises(TranslateError):
                translate(self.SERVER, "OK.", "en", "ru")
        self.assertEqual(up.call_count, 1)

    def test_malformed_json_raises_translate_error(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            resp = mock.MagicMock()
            resp.read.return_value = b"not json"
            resp.__enter__.return_value = resp
            up.return_value = resp
            with self.assertRaises(TranslateError):
                translate(self.SERVER, "OK.", "en", "ru")

    def test_empty_content_raises_translate_error(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._ok_response("   ")
            with self.assertRaises(TranslateError):
                translate(self.SERVER, "OK.", "en", "ru")

    def test_missing_choices_raises_translate_error(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            resp = mock.MagicMock()
            resp.read.return_value = json.dumps({"choices": []}).encode()
            resp.__enter__.return_value = resp
            up.return_value = resp
            with self.assertRaises(TranslateError):
                translate(self.SERVER, "OK.", "en", "ru")


class TestFindRouterModel(unittest.TestCase):
    SERVER = "http://127.0.0.1:8087"

    def _models_ok(self, data: list) -> mock.MagicMock:
        resp = mock.MagicMock()
        resp.read.return_value = json.dumps({"data": data}).encode()
        resp.__enter__.return_value = resp
        return resp

    def test_matches_translategemma_id(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._models_ok([
                {"id": "gemma-4-12b-it-Q4_0", "aliases": []},
                {"id": "mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL",
                 "aliases": ["trans"]},
            ])
            self.assertEqual(
                find_router_model(self.SERVER),
                "mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL",
            )

    def test_matches_trans_alias_when_id_differs(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._models_ok([
                {"id": "some-renamed-model", "aliases": ["trans"]},
            ])
            self.assertEqual(find_router_model(self.SERVER),
                             "some-renamed-model")

    def test_none_when_server_unreachable(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.side_effect = URLError("refused")
            self.assertIsNone(find_router_model(self.SERVER))

    def test_none_when_no_translategemma_hosted(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = self._models_ok([
                {"id": "gemma-4-12b-it-Q4_0", "aliases": ["g12"]},
            ])
            self.assertIsNone(find_router_model(self.SERVER))

    def test_none_on_malformed_body(self):
        with mock.patch("trans.urllib.request.urlopen") as up:
            resp = mock.MagicMock()
            resp.read.return_value = b"not json"
            resp.__enter__.return_value = resp
            up.return_value = resp
            self.assertIsNone(find_router_model(self.SERVER))


class TestResolveBackend(unittest.TestCase):
    def _args(self, **overrides):
        args = parse_args(["hi"])
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_prefers_running_router_with_translategemma(self):
        model_id = "mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL"
        with mock.patch(
                "trans.find_router_model", return_value=model_id,
        ) as find, mock.patch("trans.ensure_server") as ensure:
            url, model = resolve_backend(self._args(), {})
        self.assertEqual(url, "http://127.0.0.1:8087")
        self.assertEqual(
            model, "mradermacher/translategemma-4b-it-i1-GGUF:IQ4_NL"
        )
        find.assert_called_once_with(
            "http://127.0.0.1:8087", debug=False, stderr=mock.ANY
        )
        ensure.assert_not_called()

    def test_falls_back_to_dedicated_server_without_router_model(self):
        with mock.patch("trans.find_router_model", return_value=None), \
                mock.patch(
                    "trans.ensure_server", return_value="http://127.0.0.1:8144"
                ) as ensure:
            url, model = resolve_backend(self._args(), {})
        self.assertEqual(url, "http://127.0.0.1:8144")
        self.assertIsNone(model)
        ensure.assert_called_once_with(
            "127.0.0.1", 8144, True, debug=False
        )

    def test_fallback_respects_auto_start_env(self):
        with mock.patch("trans.find_router_model", return_value=None), \
                mock.patch(
                    "trans.ensure_server", return_value="http://127.0.0.1:8144"
                ) as ensure:
            resolve_backend(self._args(), {"TRANS_AUTO_START": "0"})
        ensure.assert_called_once_with(
            "127.0.0.1", 8144, False, debug=False
        )

    def test_fallback_uses_host_port_overrides(self):
        with mock.patch("trans.find_router_model", return_value=None), \
                mock.patch(
                    "trans.ensure_server", return_value="http://127.0.0.1:9000"
                ) as ensure:
            resolve_backend(
                self._args(host="127.0.0.1", port=9000), {}
            )
        ensure.assert_called_once_with(
            "127.0.0.1", 9000, True, debug=False
        )

    def test_env_url_pins_server_and_probes_it_for_a_model(self):
        with mock.patch(
                "trans.find_router_model", return_value="trans-model"
        ) as find, mock.patch("trans.ensure_server") as ensure:
            url, model = resolve_backend(
                self._args(), {"TRANS_SERVER_URL": "http://elsewhere:9999"}
            )
        self.assertEqual(url, "http://elsewhere:9999")
        self.assertEqual(model, "trans-model")
        find.assert_called_once_with(
            "http://elsewhere:9999", debug=False, stderr=mock.ANY
        )
        ensure.assert_not_called()


class TestVoiceForLanguage(unittest.TestCase):
    def test_preferred_voice_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            piper_dir = Path(tmp)
            preferred = piper_dir / "en_US-libritts_r-medium.onnx"
            preferred.write_bytes(b"x")
            with mock.patch.object(trans, "PIPER_DIR", piper_dir):
                self.assertEqual(voice_for_language("en"), str(preferred))

    def test_local_glob_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            piper_dir = Path(tmp)
            other = piper_dir / "de_DE-karlsson-low.onnx"
            other.write_bytes(b"x")
            with mock.patch.object(trans, "PIPER_DIR", piper_dir):
                self.assertEqual(voice_for_language("de"), str(other))

    def test_downloads_first_matching_voice(self):
        with tempfile.TemporaryDirectory() as tmp:
            piper_dir = Path(tmp)
            listing = (
                "en_US-amy-medium\nes_AR-daniela-high\nes_ES-sharvard-medium\n"
            )
            downloaded = piper_dir / "es_AR-daniela-high.onnx"
            downloaded.write_bytes(b"x")

            def fake_run(cmd, **kwargs):
                if cmd == [trans.DOWNLOAD_VOICES]:
                    result = mock.MagicMock()
                    result.returncode = 0
                    result.stdout = listing
                    return result
                if "es_AR-daniela-high" in cmd:
                    result = mock.MagicMock()
                    result.returncode = 0
                    return result
                raise AssertionError(f"unexpected cmd: {cmd}")

            with mock.patch.object(trans, "PIPER_DIR", piper_dir), \
                    mock.patch(
                        "trans.subprocess.run", side_effect=fake_run
                    ):
                self.assertEqual(
                    voice_for_language("es"), str(downloaded)
                )

    def test_returns_none_when_no_voice_anywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = mock.MagicMock()
            result.returncode = 0
            result.stdout = "en_US-amy-medium\n"
            with mock.patch.object(trans, "PIPER_DIR", Path(tmp)), \
                    mock.patch(
                        "trans.subprocess.run", return_value=result
                    ):
                self.assertIsNone(voice_for_language("xx"))


class TestSpeak(unittest.TestCase):
    def test_piper_plays_directly_without_output_file(self):
        # piper speaks by itself now: no -f, no player process
        with mock.patch(
            "trans.voice_for_language", return_value="/fake/ru_RU-x.onnx"
        ) as vf, mock.patch("trans.subprocess.Popen") as popen:
            popen.return_value.wait.return_value = 0
            self.assertTrue(speak("Привет!", "ru"))
        vf.assert_called_once_with("ru", debug=False, stderr=None)
        argv = popen.call_args[0][0]
        self.assertEqual(argv[0], "piper")
        self.assertIn("--cuda", argv)
        self.assertEqual(
            argv[argv.index("--model") + 1], "/fake/ru_RU-x.onnx"
        )
        self.assertNotIn("-f", argv)

    def test_feeds_text_to_piper_stdin(self):
        with mock.patch(
            "trans.voice_for_language", return_value="/fake/v.onnx"
        ), mock.patch("trans.subprocess.Popen") as popen:
            popen.return_value.wait.return_value = 0
            self.assertTrue(speak("text here", "es"))
        popen.return_value.stdin.write.assert_called_once_with(
            b"text here"
        )
        popen.return_value.stdin.close.assert_called_once()

    def test_returns_false_without_voice(self):
        with mock.patch("trans.voice_for_language", return_value=None), \
                mock.patch("sys.stderr"):
            self.assertFalse(speak("hi", "xx"))

    def test_returns_false_when_piper_fails(self):
        with mock.patch(
            "trans.voice_for_language", return_value="/fake/v.onnx"
        ), mock.patch("trans.subprocess.Popen") as popen, \
                mock.patch("sys.stderr"):
            popen.return_value.wait.return_value = 1
            self.assertFalse(speak("hi", "ru"))

    def test_piper_stderr_suppressed_unless_debug(self):
        # without --debug piper's noisy onnxruntime warnings are dropped;
        # with --debug they flow through so problems are diagnosable
        for debug, expect_capture in ((False, True), (True, False)):
            with self.subTest(debug=debug):
                with mock.patch(
                    "trans.voice_for_language",
                    return_value="/fake/v.onnx",
                ), mock.patch("trans.subprocess.Popen") as popen:
                    popen.return_value.wait.return_value = 0
                    self.assertTrue(
                        speak("hi", "ru", debug=debug)
                    )
                kwargs = popen.call_args[1]
                if expect_capture:
                    self.assertEqual(
                        kwargs["stderr"], subprocess.DEVNULL
                    )
                else:
                    self.assertIsNone(kwargs["stderr"])

    def test_returns_false_on_oserror(self):
        with mock.patch(
            "trans.voice_for_language", return_value="/fake/v.onnx"
        ), mock.patch(
            "trans.subprocess.Popen", side_effect=OSError("boom")
        ), mock.patch("sys.stderr"):
            self.assertFalse(speak("hi", "ru"))


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
        self.assertEqual(cmd[cmd.index("-hf") + 1], trans.MODEL_PATH)
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

    def test_acquire_start_lock_closes_fd_on_contention(self):
        # loser must not leak the lock fd while waiting for the other
        # instance
        with tempfile.TemporaryDirectory() as state:
            state_dir = Path(state)
            holder = trans._acquire_start_lock(state_dir)
            self.assertIsNotNone(holder)
            try:
                before = len(os.listdir("/proc/self/fd"))
                for _ in range(3):
                    self.assertIsNone(
                        trans._acquire_start_lock(state_dir)
                    )
                after = len(os.listdir("/proc/self/fd"))
                self.assertEqual(before, after)
            finally:
                os.close(holder)

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
            if sig == 0:
                raise ProcessLookupError()  # gone after SIGTERM
            return None

        with tempfile.TemporaryDirectory() as state:
            state_dir = Path(state)
            pid_file = state_dir / "server.pid"
            pid_file.write_text("1234\nllama-server -m x\n")
            with mock.patch("trans.os.kill", side_effect=record_kill), \
                    mock.patch("trans.time.sleep"):
                self.assertTrue(stop_server(state_dir))
            self.assertFalse(pid_file.exists())
        self.assertEqual(kill_calls[0][0], 1234)

    def test_stop_server_missing_pidfile_returns_false(self):
        with mock.patch.object(Path, "exists", return_value=False):
            self.assertFalse(stop_server(Path("/tmp/fake-state")))


class TestCLI(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = parse_args(["hello", "world"])
        self.assertEqual(args.phrase, ["hello", "world"])
        self.assertTrue(args.speak_flag)
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8144)
        self.assertFalse(args.stop)
        self.assertIsNone(args.from_lang)
        self.assertIsNone(args.to_lang)

    def test_parse_args_from_to(self):
        args = parse_args(["--from", "Russian", "--to", "Spanish", "hi"])
        self.assertEqual(args.from_lang, "ru")
        self.assertEqual(args.to_lang, "es")

    def test_parse_args_from_accepts_code(self):
        args = parse_args(["--from", "fr", "hi"])
        self.assertEqual(args.from_lang, "fr")

    def test_parse_args_rejects_unknown_language(self):
        with self.assertRaises(SystemExit), \
                mock.patch("sys.stderr"):
            parse_args(["--to", "klingon", "hi"])

    def test_parse_args_no_speak(self):
        self.assertFalse(parse_args(["--no-speak", "hi"]).speak_flag)

    def test_parse_args_port(self):
        self.assertEqual(parse_args(["--port", "9000", "hi"]).port, 9000)

    def test_parse_args_stop_server(self):
        self.assertTrue(parse_args(["--stop-server"]).stop)

    def test_parse_args_empty_phrase_is_repl(self):
        self.assertEqual(parse_args([]).phrase, [])

    def test_parse_args_debug_flag(self):
        self.assertTrue(parse_args(["--debug", "hi"]).debug)
        self.assertFalse(parse_args(["hi"]).debug)

    def test_parse_args_rejects_unknown_flag(self):
        with self.assertRaises(SystemExit), \
                mock.patch("sys.stderr"):
            parse_args(["--bogus", "hi"])

    def test_parse_args_router_flags(self):
        args = parse_args(["--router-port", "9001", "hi"])
        self.assertEqual(args.router_host, "127.0.0.1")
        self.assertEqual(args.router_port, 9001)
        args = parse_args(["--router-host", "localhost", "hi"])
        self.assertEqual(args.router_host, "localhost")
        self.assertEqual(args.router_port, 8087)

    def test_translate_once_prints_output_language_prefix(self):
        out = io.StringIO()
        with mock.patch("trans.detect_language", return_value="en"), \
                mock.patch("trans.translate", return_value="¡Hola!"), \
                mock.patch("trans.speak", return_value=True) as sp:
            translate_once(
                "Hello!", "http://x", None, None, True, stdout=out
            )
        self.assertEqual(out.getvalue(), "es: ¡Hola!\n")
        sp.assert_called_once_with("¡Hola!", "es", debug=False, stderr=None)

    def test_translate_once_english_defaults_to_spanish(self):
        out = io.StringIO()
        with mock.patch("trans.detect_language", return_value="en"), \
                mock.patch("trans.translate", return_value="Hola") as tr, \
                mock.patch("trans.speak"):
            translate_once(
                "Hello!", "http://x", None, None, True, stdout=out
            )
        tr.assert_called_once_with(
            "http://x", "Hello!", "en", "es",
            debug=False, stderr=None, model=None,
        )

    def test_translate_once_foreign_defaults_to_english(self):
        out = io.StringIO()
        with mock.patch("trans.detect_language", return_value="fr"), \
                mock.patch("trans.translate", return_value="Hello") as tr, \
                mock.patch("trans.speak"):
            translate_once(
                "Bonjour", "http://x", None, None, True, stdout=out
            )
        tr.assert_called_once_with(
            "http://x", "Bonjour", "fr", "en",
            debug=False, stderr=None, model=None,
        )

    def test_translate_once_explicit_from_skips_detection(self):
        out = io.StringIO()
        with mock.patch("trans.detect_language") as det, \
                mock.patch("trans.translate", return_value="Hello") as tr, \
                mock.patch("trans.speak"):
            translate_once(
                "Bonjour", "http://x", "fr", "de", True, stdout=out
            )
        det.assert_not_called()
        tr.assert_called_once_with(
            "http://x", "Bonjour", "fr", "de",
            debug=False, stderr=None, model=None,
        )

    def test_translate_once_blank_input_noop(self):
        out = io.StringIO()
        with mock.patch("trans.translate") as tr:
            translate_once("   ", "http://x", None, None, True, stdout=out)
        tr.assert_not_called()
        self.assertEqual(out.getvalue(), "")

    def test_run_repl_translates_lines_until_quit(self):
        out = io.StringIO()
        with mock.patch("trans.detect_language", return_value="en"), \
                mock.patch("trans.translate", return_value="Hola"), \
                mock.patch("trans.speak", return_value=True), \
                mock.patch("builtins.input",
                           side_effect=["Hello!", "   ", "q"]):
            run_repl("http://x", None, None, True, stdout=out)
        self.assertEqual(out.getvalue().count("es: Hola\n"), 1)

    def test_run_repl_survives_translate_error(self):
        out = io.StringIO()
        with mock.patch("trans.detect_language", return_value="en"), \
                mock.patch("trans.translate",
                           side_effect=TranslateError("boom")), \
                mock.patch("trans.speak"), \
                mock.patch("builtins.input",
                           side_effect=["Hello!", "q"]), \
                mock.patch("sys.stderr"):
            run_repl("http://x", None, None, True, stdout=out)

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
        with mock.patch("trans.find_router_model", return_value=None), \
                mock.patch("trans.ensure_server", return_value="http://x"), \
                mock.patch("trans.detect_language", return_value="en"), \
                mock.patch("trans.translate", return_value="¡Hola!"), \
                mock.patch("trans.speak", return_value=True), \
                mock.patch("sys.stdout"):
            code = main(["Hello!"])
        self.assertEqual(code, 0)

    def test_main_one_shot_with_from_to(self):
        with mock.patch("trans.find_router_model", return_value=None), \
                mock.patch("trans.ensure_server", return_value="http://x"), \
                mock.patch("trans.detect_language") as det, \
                mock.patch("trans.translate", return_value="Привет"), \
                mock.patch("trans.speak", return_value=True), \
                mock.patch("sys.stdout"):
            code = main(["--from", "English", "--to", "Russian", "Hello!"])
        self.assertEqual(code, 0)
        det.assert_not_called()

    def test_main_one_shot_error_exits_nonzero(self):
        with mock.patch("trans.find_router_model", return_value=None), \
                mock.patch("trans.ensure_server", return_value="http://x"), \
                mock.patch("trans.detect_language", return_value="en"), \
                mock.patch("trans.translate",
                           side_effect=TranslateError("bad")), \
                mock.patch("sys.stderr"), \
                mock.patch("sys.stdout"):
            self.assertEqual(main(["Hello!"]), 1)

    def test_main_fatal_error_returns_one(self):
        with mock.patch("trans.find_router_model", return_value=None), \
                mock.patch(
                    "trans.ensure_server",
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
        with mock.patch.dict(
                os.environ,
                {"TRANS_SERVER_URL": "http://elsewhere:9999"}), \
                mock.patch("trans.ensure_server") as ensure, \
                mock.patch("trans.urllib.request.urlopen") as up, \
                mock.patch("sys.stdout"), \
                mock.patch("trans.speak", return_value=True):
            resp = mock.MagicMock()
            resp.read.return_value = ok_body
            resp.__enter__.return_value = resp
            up.return_value = resp
            self.assertEqual(main(["--from", "en", "Hello!"]), 0)
        ensure.assert_not_called()
        self.assertEqual(
            up.call_args[0][0].full_url,
            "http://elsewhere:9999/v1/chat/completions",
        )


class TestDebug(unittest.TestCase):
    """--debug echoes every external command and HTTP request."""

    def test_start_server_detached_prints_shell_command(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as state, \
                mock.patch("trans.STATE_DIR", Path(state)), \
                mock.patch("trans.subprocess.Popen") as popen:
            popen.return_value.pid = 4242
            trans._start_server_detached(
                "127.0.0.1", 8144, Path(state), debug=True, stderr=err
            )
        self.assertIn("llama-server", err.getvalue())
        self.assertIn("--chat-template-file", err.getvalue())
        self.assertIn("--temp 0", err.getvalue())

    def test_start_server_detached_quiet_by_default(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as state, \
                mock.patch("trans.STATE_DIR", Path(state)), \
                mock.patch("trans.subprocess.Popen") as popen:
            popen.return_value.pid = 4242
            trans._start_server_detached(
                "127.0.0.1", 8144, Path(state), debug=False, stderr=err
            )
        self.assertEqual(err.getvalue(), "")

    def test_speak_prints_piper_command(self):
        err = io.StringIO()
        with mock.patch(
            "trans.voice_for_language", return_value="/fake/v.onnx"
        ), mock.patch("trans.subprocess.Popen") as popen:
            popen.return_value.wait.return_value = 0
            self.assertTrue(speak("hi", "ru", debug=True, stderr=err))
        argv = popen.call_args[0][0]
        printed = err.getvalue()
        self.assertIn("piper", printed)
        self.assertIn("--cuda", printed)
        self.assertIn("/fake/v.onnx", printed)
        self.assertIn("<<< hi", printed)  # stdin text shown as heredoc
        self.assertEqual(argv[-1], "/fake/v.onnx")

    def test_voice_download_prints_download_commands(self):
        err = io.StringIO()
        listing = mock.MagicMock()
        listing.returncode = 0
        listing.stdout = "es_ES-davefx-medium\n"
        download = mock.MagicMock()
        download.returncode = 0
        target_dir_box: list = []

        def fake_run(cmd, **kwargs):
            # listing first, then the download makes the file appear
            if cmd == [trans.DOWNLOAD_VOICES]:
                return listing
            (target_dir_box[0] / "es_ES-davefx-medium.onnx").write_bytes(
                b"x"
            )
            return download

        with tempfile.TemporaryDirectory() as tmp:
            target_dir_box.append(Path(tmp))
            with mock.patch.object(trans, "PIPER_DIR", Path(tmp)), \
                    mock.patch(
                        "trans.subprocess.run", side_effect=fake_run
                    ):
                voice = voice_for_language("es", debug=True, stderr=err)
        self.assertEqual(
            voice, str(Path(tmp) / "es_ES-davefx-medium.onnx")
        )
        printed = err.getvalue()
        self.assertIn("download_voices", printed)
        self.assertIn("--download-dir", printed)
        self.assertIn("es_ES-davefx-medium", printed)

    def test_chat_request_printed_in_debug(self):
        err = io.StringIO()
        ok = mock.MagicMock()
        ok.read.return_value = json.dumps(
            {"choices": [{"message": {"role": "assistant",
                                      "content": "es"}}]}
        ).encode()
        ok.__enter__.return_value = ok
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = ok
            code = detect_language(
                "http://127.0.0.1:8144", "Hola", debug=True, stderr=err
            )
        self.assertEqual(code, "es")
        printed = err.getvalue()
        self.assertIn("POST http://127.0.0.1:8144/v1/chat/completions",
                      printed)
        self.assertIn("Hola", printed)

    def test_debug_disabled_no_chat_print(self):
        err = io.StringIO()
        ok = mock.MagicMock()
        ok.read.return_value = json.dumps(
            {"choices": [{"message": {"role": "assistant",
                                      "content": "es"}}]}
        ).encode()
        ok.__enter__.return_value = ok
        with mock.patch("trans.urllib.request.urlopen") as up:
            up.return_value = ok
            detect_language(
                "http://127.0.0.1:8144", "Hola", debug=False, stderr=err
            )
        self.assertEqual(err.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
