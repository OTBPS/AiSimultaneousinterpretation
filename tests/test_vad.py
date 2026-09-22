"""Regression test for the Silero v5 64-sample context requirement.

Without the context prepend the model silently returns ~0.0006 on loud speech
instead of raising, which is how this bug got into the first implementation.
"""
import numpy as np
import soundfile as sf
import pytest

from app.vad import SileroVad, FRAME_SAMPLES

MODEL = r"D:\AI\Models\silero_vad.onnx"
WAV = "bench/speech.wav"


def _probs(n=600):
    audio, sr = sf.read(WAV, dtype="float32")
    assert sr == 16000
    v = SileroVad(MODEL)
    return np.array([v(audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
                     for i in range(n)])


def test_detects_speech_in_speech_clip():
    p = _probs()
    assert p.max() > 0.9, "VAD sees no speech at all -- context prepend broken?"
    assert (p > 0.5).mean() > 0.5, "continuous speech should be mostly voiced"


def test_silence_is_not_speech():
    v = SileroVad(MODEL)
    silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    assert max(v(silence) for _ in range(20)) < 0.2


def test_reset_clears_context():
    v = SileroVad(MODEL)
    audio, _ = sf.read(WAV, dtype="float32")
    first = v(audio[:FRAME_SAMPLES])
    for i in range(1, 50):
        v(audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
    v.reset()
    assert v(audio[:FRAME_SAMPLES]) == pytest.approx(first, abs=1e-6)


def test_wrong_frame_size_raises():
    v = SileroVad(MODEL)
    with pytest.raises(ValueError):
        v(np.zeros(256, dtype=np.float32))
