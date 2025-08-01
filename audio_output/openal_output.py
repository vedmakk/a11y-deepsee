from __future__ import annotations

import math
import tempfile
import threading
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    # The canonical PyPI package is simply called "openal" (PyOpenAL bindings)
    from openal import oalInit, oalQuit, oalOpen, Listener  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover – optional dependency
    raise ImportError(
        "PyOpenAL is required for OpenALAudioOutput. Install with `pip install openal`."
    ) from exc

from .base import AudioOutput

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# (x, y, z, amplitude, frequency)
Source3D = Tuple[float, float, float, float, float]


class OpenALAudioOutput(AudioOutput):
    MAX_SOURCES = 64  # practical limit to avoid OpenAL running out of voices
    """True 3-D audio output using OpenAL.

    The class keeps a pool of looping OpenAL sources – one per active sound in the scene – and
    continuously updates their *position*, *gain* (volume) and *pitch* according to the values
    provided by the depth→audio mapper.

    Notes
    -----
    • **PyOpenAL** must be available (``pip install openal``).  
    • A short sine-wave sample is generated at runtime and reused for *all* sources.  The sample’s
      base frequency is 440 Hz, *pitch* controls are used to reach arbitrary frequencies.
    • Distance attenuation can be handled by OpenAL itself but, for consistency with the existing
      codebase, we directly control *gain* here so that nearer objects are louder.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        sample_seconds: float = 0.25,
        base_freq: float = 440.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.sample_seconds = sample_seconds
        self.base_freq = base_freq

        # Lazily initialised in ``start()``
        self._listener: Listener | None = None

        self._lock = threading.Lock()
        self._sources: List = []  # Pool of currently active OpenAL sources

        # Pre-generate a small WAV file containing a sine wave at *base_freq* that we can reuse for
        # every OpenAL source.  Using a temporary file is the simplest way to hand audio data to
        # PyOpenAL without additional dependencies.
        self._sample_path = Path(self._generate_sample())

    # ------------------------------------------------------------------
    # AudioOutput interface
    # ------------------------------------------------------------------
    def start(self) -> None:  # noqa: D401 – keep interface naming consistent
        """Initialise OpenAL and start playback."""
        oalInit()
        # Ensure there *is* a listener and that it sits at the origin looking along the −Z axis
        self._listener = Listener()
        self._listener.set_position([0, 0, 0])
        # Some PyOpenAL forks expect a 6-element tuple, others separate "at"/"up" kwargs.
        try:
            self._listener.set_orientation([0, 0, -1, 0, 1, 0])
        except TypeError:
            self._listener.set_orientation(at=[0, 0, -1], up=[0, 1, 0])

    def stop(self) -> None:  # noqa: D401 – keep interface naming consistent
        """Stop playback and release **all** OpenAL resources."""
        with self._lock:
            for src in self._sources:
                src.stop()
                # Different PyOpenAL forks use either `destroy()` or `delete()` – handle both.
                if hasattr(src, "destroy"):
                    src.destroy()
                elif hasattr(src, "delete"):
                    src.delete()
            self._sources.clear()

        oalQuit()  # Shut down the device/context – safe to call multiple times

    def update_sources(self, sources: List[Source3D]) -> None:  # noqa: D401
        """Synchronise the internal source pool with *sources* provided by the mapper."""
        with self._lock:
            # ------------------------------------------------------------------
            # Limit number of simultaneous voices to avoid AL_OUT_OF_MEMORY
            # ------------------------------------------------------------------
            if len(sources) > self.MAX_SOURCES:
                # Keep the loudest sources (gain index 3)
                sources = sorted(sources, key=lambda s: s[3], reverse=True)[: self.MAX_SOURCES]

            # ------------------------------------------------------------------
            # Ensure we have enough OpenAL sources
            # ------------------------------------------------------------------
            while len(self._sources) < len(sources):
                src = oalOpen(str(self._sample_path))
                src.set_looping(True)
                src.play()
                self._sources.append(src)

            # Remove surplus sources
            for extra in self._sources[len(sources) :]:
                extra.stop()
                extra.destroy()
            self._sources = self._sources[: len(sources)]

            # ------------------------------------------------------------------
            # Update active sources
            # ------------------------------------------------------------------
            for src, (x, y, z, gain, freq) in zip(self._sources, sources):
                src.set_position([x, y, z])
                src.set_gain(max(0.0, min(1.0, gain)))  # Clamp just in case
                # OpenAL’s pitch parameter is a *multiplier* of the buffer’s base frequency.
                src.set_pitch(freq / self.base_freq if self.base_freq else 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _generate_sample(self) -> str:
        """Create a temporary mono WAV file with a sine wave at *base_freq*.

        The file is **not** deleted automatically because it is kept open by OpenAL for the whole
        lifetime of the process.  It will be removed by the OS eventually.
        """
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        duration_samples = int(self.sample_rate * self.sample_seconds)
        t = np.linspace(0, self.sample_seconds, duration_samples, endpoint=False)
        # Use 16-bit PCM – OpenAL handles this format natively
        signal = (np.sin(2 * math.pi * self.base_freq * t) * 0.5 * (2**15 - 1)).astype(np.int16)

        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(signal.tobytes())

        return tmp.name
