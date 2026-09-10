"""Private-pilot Qwen3-TTS WebSocket service for Bible Reading."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
from collections.abc import Generator

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from mlx_audio.tts.utils import load_model

os.environ["PATH"] = f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
TEMPERATURE = 0.6
STREAMING_INTERVAL = 0.32
MAX_TEXT_CHARS = 12_000
MAX_SYNTHESIS_CHARS = 360
APP = FastAPI()
LOGGER = logging.getLogger("bible-reading-tts")
MODEL = None
REFERENCE_TEXT = ""
MODEL_LOCK = threading.Lock()


def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if os.environ.get("BIBLE_READING_TTS_LOAD_ENV_FILE", "true").lower() == "true":
    load_env_file(os.path.join(os.path.dirname(__file__), ".env"))


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


def load_runtime() -> None:
    global MODEL, REFERENCE_TEXT
    if MODEL is not None:
        return
    with open(required("TTS_REFERENCE_TRANSCRIPT"), encoding="utf-8") as source:
        REFERENCE_TEXT = source.read().strip()
    MODEL = load_model(MODEL_ID)


def b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def verify_playback_token(token: str, text: str) -> bool:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        expected = b64url(hmac.new(required("TTS_PLAYBACK_SIGNING_SECRET").encode(), encoded_payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(encoded_signature, expected):
            return False
        payload = json.loads(base64.urlsafe_b64decode(encoded_payload + "=" * (-len(encoded_payload) % 4)))
        return (isinstance(payload.get("exp"), (int, float)) and payload["exp"] > time.time()
                and hmac.compare_digest(payload.get("textDigest", ""), hashlib.sha256(text.encode()).hexdigest())
                and isinstance(payload.get("nonce"), str))
    except (ValueError, TypeError, json.JSONDecodeError):
        return False


def pcm16(samples: object) -> bytes:
    values = np.asarray(samples, dtype=np.float32)
    return (np.clip(values, -1, 1) * 32767).astype("<i2", copy=False).tobytes()


def synthesis_chunks(text: str) -> Generator[str, None, None]:
    remaining = text.strip()
    while remaining:
        if len(remaining) <= MAX_SYNTHESIS_CHARS:
            yield remaining
            return
        window = remaining[: MAX_SYNTHESIS_CHARS + 1]
        boundaries = [match.end() for match in re.finditer(r"[.!?。！？]\s*", window)]
        split_at = boundaries[-1] if boundaries else window.rfind(" ")
        if split_at <= 0:
            split_at = MAX_SYNTHESIS_CHARS
        yield remaining[:split_at].strip()
        remaining = remaining[split_at:].strip()


def packet_measurement(sequence: int, sample_rate: int, audio: bytes, elapsed_seconds: float) -> dict[str, float | int]:
    """Return content-free PCM facts suitable for cross-process timing logs."""
    return {
        "sequence": sequence,
        "bytes": len(audio),
        "duration": round(len(audio) / (sample_rate * 2), 3),
        "elapsed": round(elapsed_seconds, 3),
    }


def generate(text: str, cancelled: threading.Event) -> Generator[tuple[int, bytes], None, None]:
    assert MODEL is not None
    with MODEL_LOCK:
        for chunk_index, chunk in enumerate(synthesis_chunks(text)):
            started_at = time.perf_counter()
            first_packet = True
            for result in MODEL.generate(text=chunk, ref_audio=required("TTS_REFERENCE_AUDIO"), ref_text=REFERENCE_TEXT, temperature=TEMPERATURE, stream=True, streaming_interval=STREAMING_INTERVAL):
                if cancelled.is_set():
                    return
                if first_packet:
                    LOGGER.warning("TTS timing chunk=%d first_packet=%.3fs", chunk_index, time.perf_counter() - started_at)
                    first_packet = False
                yield result.sample_rate, pcm16(result.audio)


@APP.on_event("startup")
async def startup() -> None:
    load_runtime()


@APP.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@APP.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        request = await asyncio.wait_for(websocket.receive_json(), timeout=15)
        text = request.get("text", "") if isinstance(request, dict) else ""
        token = request.get("token", "") if isinstance(request, dict) else ""
        valid = isinstance(text, str) and bool(text.strip()) and len(text) <= MAX_TEXT_CHARS and isinstance(token, str) and verify_playback_token(token, text)
        if not valid:
            await websocket.send_json({"type": "error", "code": "unauthorized"})
            await websocket.close(code=1008)
            return
        cancelled = threading.Event()
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        def producer() -> None:
            try:
                started_at = time.perf_counter()
                sequence = 0
                for packet in generate(text, cancelled):
                    sequence += 1
                    sample_rate, audio = packet
                    measurement = packet_measurement(sequence, sample_rate, audio, time.perf_counter() - started_at)
                    LOGGER.warning(
                        "TTS generated sequence=%(sequence)d bytes=%(bytes)d duration=%(duration).3fs elapsed=%(elapsed).3fs",
                        measurement,
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, ("audio", packet))
                loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
            except Exception:
                LOGGER.exception("TTS generation failed")
                loop.call_soon_threadsafe(queue.put_nowait, ("error", None))
        threading.Thread(target=producer, daemon=True).start()
        sent_metadata = False
        sent_sequence = 0
        sent_bytes = 0
        stream_started_at = time.perf_counter()
        while True:
            kind, value = await queue.get()
            if kind == "audio":
                sample_rate, audio = value  # type: ignore[misc]
                if not sent_metadata:
                    await websocket.send_json({"type": "metadata", "format": "pcm_s16le", "sampleRate": sample_rate, "channels": 1})
                    sent_metadata = True
                sent_sequence += 1
                sent_bytes += len(audio)
                measurement = packet_measurement(sent_sequence, sample_rate, audio, time.perf_counter() - stream_started_at)
                measurement["total_bytes"] = sent_bytes
                LOGGER.warning(
                    "TTS sent sequence=%(sequence)d bytes=%(bytes)d duration=%(duration).3fs elapsed=%(elapsed).3fs total_bytes=%(total_bytes)d",
                    measurement,
                )
                await websocket.send_bytes(audio)
            elif kind == "end":
                await websocket.send_json({"type": "end"})
                return
            else:
                await websocket.send_json({"type": "error", "code": "generation_failed"})
                return
    except (WebSocketDisconnect, asyncio.TimeoutError):
        return
    finally:
        if "cancelled" in locals():
            cancelled.set()
