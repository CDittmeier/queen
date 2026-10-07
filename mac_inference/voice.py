"""ElevenLabs text-to-speech with character timings for the explanation voice-over."""

import json
import os
import ssl
import threading
import urllib.error
import urllib.request
from collections import OrderedDict

API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps"
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # "George", a calm narration voice.
DEFAULT_MODEL = "eleven_multilingual_v2"
MAX_CHARACTERS = 5000
CACHE_SIZE = 64


class VoiceError(Exception):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def load_env(path):
    """Read KEY=value lines into the environment without overriding it."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _tls_context():
    # Python.org builds ship without a CA bundle; certifi is already installed.
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


class Voice:
    def __init__(self, api_key, voice_id=DEFAULT_VOICE, model_id=DEFAULT_MODEL):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self._cache = OrderedDict()  # Repeated text is free to replay.
        self._lock = threading.Lock()
        self._tls = _tls_context()

    @classmethod
    def from_env(cls):
        return cls(
            os.environ.get("ELEVENLABS_API_KEY", "").strip(),
            os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE,
            os.environ.get("ELEVENLABS_MODEL_ID") or DEFAULT_MODEL,
        )

    @property
    def enabled(self):
        return bool(self.api_key)

    def speak(self, text):
        """Return MP3 audio (base64) and per-character start/end times."""
        if not self.enabled:
            raise VoiceError("Add ELEVENLABS_API_KEY to .env to enable voice.", 503)
        if not isinstance(text, str) or not text.strip():
            raise VoiceError("Nothing to read.", 400)
        if len(text) > MAX_CHARACTERS:
            raise VoiceError("That passage is too long to read.", 400)
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
        speech = self._request(text)
        with self._lock:
            self._cache[text] = speech
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return speech

    def _request(self, text):
        request = urllib.request.Request(
            API.format(voice=self.voice_id) + "?output_format=mp3_44100_128",
            data=json.dumps({"text": text, "model_id": self.model_id}).encode(),
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60, context=self._tls) as r:
                answer = json.load(r)
        except urllib.error.HTTPError as error:
            raise VoiceError(f"ElevenLabs: {_detail(error)}") from None
        except (urllib.error.URLError, TimeoutError) as error:
            raise VoiceError(f"Couldn't reach ElevenLabs ({error}).") from None
        alignment = answer.get("alignment") or {}
        return {
            "audio": answer["audio_base64"],
            "characters": alignment.get("characters", []),
            "starts": alignment.get("character_start_times_seconds", []),
            "ends": alignment.get("character_end_times_seconds", []),
        }


def _detail(error):
    try:
        detail = json.load(error).get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, dict):
        detail = detail.get("message")
    return detail or f"HTTP {error.code}"
