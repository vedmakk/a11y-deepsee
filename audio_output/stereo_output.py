from __future__ import annotations

import math
import threading
from typing import List, Tuple

import numpy as np
import sounddevice as sd

Source = Tuple[float, float, float]  # azimuth (-1..1), amplitude, frequency


class StereoAudioOutput:
    """Simple stereo renderer using panning & volume scaling.

    This is *not* a full HRTF implementation but works reasonably well with
    headphones (including AirPods). A more sophisticated implementation could
    replace this class while keeping the same public API.
    """

    def __init__(self, sample_rate: int = 44100, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.sources: List[Source] = []
        self.phase = 0.0
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            channels=2,
            dtype="float32",
            callback=self._callback,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:  # noqa: D401
        self._stream.start()

    def stop(self) -> None:  # noqa: D401
        self._stream.stop()
        self._stream.close()

    def update_sources(self, sources: List[Source]) -> None:  # noqa: D401
        with self._lock:
            self.sources = sources.copy()

    # ------------------------------------------------------------------
    # PortAudio callback
    # ------------------------------------------------------------------
    def _callback(self, outdata, frames, time, status):  # noqa: D401
        if status:
            print("Audio callback status:", status)
        t = (np.arange(frames) + self.phase) / self.sample_rate

        with self._lock:
            active = list(self.sources)

        buffer = np.zeros((frames, 2), dtype=np.float32)
        for azimuth, amp, freq in active:
            signal = np.sin(2 * math.pi * freq * t) * amp
            left = signal * (1.0 - azimuth) * 0.5
            right = signal * (1.0 + azimuth) * 0.5
            buffer[:, 0] += left
            buffer[:, 1] += right

        # Avoid clipping
        buffer = np.clip(buffer, -1.0, 1.0)
        outdata[:] = buffer
        self.phase = (self.phase + frames) % self.sample_rate
