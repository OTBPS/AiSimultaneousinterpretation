"""Silero VAD on onnxruntime (CPU).

We drive the ONNX graph directly instead of using the `silero-vad` pip package,
because that package hard-depends on torch + torchaudio (~2.5 GB) to ship a
2 MB model. Running on CPU is deliberate: the iGPU is a contended resource
(E1: concurrent ASR+MT is 4.69x worse than serial), and VAD costs well under
1 ms per frame on CPU.

The model is stateful -- it expects strictly sequential 512-sample frames at
16 kHz and carries an LSTM state between them. `reset()` must be called
whenever the audio stream is discontinuous (source switch, pause).

Undocumented v5 detail, found the hard way: the exported graph does NOT take a
bare 512-sample frame. It expects 64 samples of the *previous* frame prepended
(576 total). Feeding a bare frame returns ~0.0006 for loud speech -- it fails
silently rather than erroring, so this is verified in tests/test_vad.py.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

FRAME_SAMPLES = 512          # Silero v5 requires exactly 512 @ 16 kHz (32 ms)
CONTEXT_SAMPLES = 64         # prepended tail of the previous frame
SAMPLE_RATE = 16000


class SileroVad:
    def __init__(self, model_path: str, threads: int = 1) -> None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = threads
        opts.intra_op_num_threads = threads
        # VAD runs on every frame; graph optimisation pays for itself.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(model_path, sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample frame."""
        if len(frame) != FRAME_SAMPLES:
            raise ValueError(f"expected {FRAME_SAMPLES} samples, got {len(frame)}")
        frame = frame.astype(np.float32, copy=False)
        inp = np.concatenate((self._context, frame)).reshape(1, -1)
        out, self._state = self._sess.run(
            None, {"input": inp, "state": self._state, "sr": self._sr},
        )
        self._context = frame[-CONTEXT_SAMPLES:].copy()
        return float(out[0][0])
