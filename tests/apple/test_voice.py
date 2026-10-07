import io
import json
import os
import urllib.error

import pytest

from mac_inference import voice as voice_module
from mac_inference.voice import Voice, VoiceError, load_env


def test_load_env_reads_values_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nELEVENLABS_API_KEY='abc'\nEXISTING=new\nnot a pair\n")
    environ = {"EXISTING": "old"}
    monkeypatch.setattr(os, "environ", environ)
    load_env(env)
    assert environ == {"ELEVENLABS_API_KEY": "abc", "EXISTING": "old"}
    load_env(tmp_path / "missing.env")  # A missing file is fine.


def test_voice_is_disabled_without_a_key():
    voice = Voice("")
    assert not voice.enabled
    with pytest.raises(VoiceError) as error:
        voice.speak("Knight to g3.")
    assert error.value.status == 503


@pytest.mark.parametrize("text", [None, "", "   ", 42, "x" * 5001])
def test_speak_rejects_bad_text(text):
    with pytest.raises(VoiceError) as error:
        Voice("key").speak(text)
    assert error.value.status == 400


def test_speak_returns_timings_and_caches(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout, context):
        calls.append(json.loads(request.data))
        assert request.get_header("Xi-api-key") == "key"
        return io.BytesIO(
            json.dumps(
                {
                    "audio_base64": "AAAA",
                    "alignment": {
                        "characters": list("Hi"),
                        "character_start_times_seconds": [0.0, 0.1],
                        "character_end_times_seconds": [0.1, 0.2],
                    },
                }
            ).encode()
        )

    monkeypatch.setattr(voice_module.urllib.request, "urlopen", fake_urlopen)
    voice = Voice("key")
    speech = voice.speak("Hi")
    assert speech == {
        "audio": "AAAA",
        "characters": ["H", "i"],
        "starts": [0.0, 0.1],
        "ends": [0.1, 0.2],
    }
    assert voice.speak("Hi") is speech
    assert calls == [{"text": "Hi", "model_id": voice.model_id}]


def test_speak_reports_service_errors(monkeypatch):
    def fake_urlopen(request, timeout, context):
        body = io.BytesIO(b'{"detail": {"message": "quota exceeded"}}')
        raise urllib.error.HTTPError(request.full_url, 401, "", {}, body)

    monkeypatch.setattr(voice_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(VoiceError, match="quota exceeded") as error:
        Voice("key").speak("Hi")
    assert error.value.status == 502
