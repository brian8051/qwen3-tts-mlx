import base64
import hashlib
import hmac
import importlib.util
import json
import os
import sys
import threading
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
APP_PATH = ROOT / "bible-reading-tts" / "app.py"


def load_app():
    os.environ["BIBLE_READING_TTS_LOAD_ENV_FILE"] = "false"
    numpy = types.ModuleType("numpy")
    numpy.float32 = object()
    numpy.asarray = lambda values, dtype=None: values
    numpy.clip = lambda values, _low, _high: values
    sys.modules["numpy"] = numpy
    fastapi = types.ModuleType("fastapi")
    class FastAPI:
        def on_event(self, _name):
            return lambda function: function
        def get(self, _path):
            return lambda function: function
        def websocket(self, _path):
            return lambda function: function
    fastapi.FastAPI = FastAPI
    fastapi.WebSocket = object
    fastapi.WebSocketDisconnect = type("WebSocketDisconnect", (Exception,), {})
    sys.modules["fastapi"] = fastapi
    utils = types.ModuleType("mlx_audio.tts.utils")
    utils.load_model = lambda _model_id: object()
    sys.modules["mlx_audio"] = types.ModuleType("mlx_audio")
    sys.modules["mlx_audio.tts"] = types.ModuleType("mlx_audio.tts")
    sys.modules["mlx_audio.tts.utils"] = utils
    spec = importlib.util.spec_from_file_location("br_tts_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


APP = load_app()


class RuntimeProtocolTests(unittest.TestCase):
    def test_natural_punctuation_is_sent_to_qwen_without_marker_or_silence_protocol(self):
        self.assertFalse(hasattr(APP, "TITLE_PAUSE_MARKER"))
        self.assertFalse(hasattr(APP, "silence_pcm"))
        self.assertEqual(list(APP.synthesis_chunks("Title, Body.")), ["Title, Body."])

    def test_generation_streams_only_model_pcm_for_natural_title_punctuation(self):
        class Result:
            sample_rate = 24_000
            audio = [0.25]

        class Model:
            def __init__(self):
                self.texts = []

            def generate(self, **kwargs):
                self.texts.append(kwargs["text"])
                yield Result()

        model = Model()
        original_pcm16 = APP.pcm16
        APP.MODEL = model
        APP.REFERENCE_TEXT = "reference"
        APP.pcm16 = lambda _audio: bytes((1, 2))
        os.environ["TTS_REFERENCE_AUDIO"] = "test-only"
        try:
            self.assertEqual(
                list(APP.generate("Title, Body.", threading.Event())),
                [(24_000, bytes((1, 2)))],
            )
            self.assertEqual(model.texts, ["Title, Body."])
        finally:
            APP.pcm16 = original_pcm16

    def test_packet_measurement_is_content_free_and_pcm_derived(self):
        measurement = APP.packet_measurement(7, 24_000, b"\0" * 15_360, 1.2346)
        self.assertEqual(measurement, {"sequence": 7, "bytes": 15_360, "duration": 0.32, "elapsed": 1.235})

    def test_signed_request_must_match_exact_text(self):
        os.environ["TTS_PLAYBACK_SIGNING_SECRET"] = "test-secret"
        text = "Authenticated request"
        payload = {"exp": time.time() + 60, "textDigest": hashlib.sha256(text.encode()).hexdigest(), "nonce": "test"}
        encoded = APP.b64url(json.dumps(payload).encode())
        signature = APP.b64url(hmac.new(b"test-secret", encoded.encode(), hashlib.sha256).digest())
        self.assertTrue(APP.verify_playback_token(f"{encoded}.{signature}", text))
        self.assertFalse(APP.verify_playback_token(f"{encoded}.{signature}", "different text"))

    def test_cancelled_generation_emits_no_audio_or_end_packet(self):
        class Result:
            sample_rate = 24_000
            audio = [0.25]
        class Model:
            def generate(self, **_kwargs):
                yield Result()
        APP.MODEL = Model()
        APP.REFERENCE_TEXT = "reference"
        os.environ["TTS_REFERENCE_AUDIO"] = "test-only"
        cancelled = threading.Event()
        cancelled.set()
        self.assertEqual(list(APP.generate("Body.", cancelled)), [])


if __name__ == "__main__":
    unittest.main()
