"""Sample management system for loading and caching WAV files."""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class AudioSample:
    """Represents a loaded audio sample with metadata."""
    
    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int,
        channels: int,
        loop: bool = True
    ):
        """Initialize audio sample.
        
        Parameters
        ----------
        data : np.ndarray
            Audio data as float32 array, shape (frames,) for mono or (frames, channels) for multi-channel
        sample_rate : int
            Sample rate in Hz
        channels : int
            Number of audio channels (1 for mono, 2 for stereo)
        loop : bool
            Whether this sample should loop seamlessly
        """
        self.data = data.astype(np.float32)
        self.sample_rate = sample_rate
        self.channels = channels
        self.loop = loop
        self.duration = len(data) / sample_rate
        
        # Ensure data is in the right shape
        if self.channels == 1 and len(self.data.shape) == 2:
            self.data = self.data.mean(axis=1)  # Convert stereo to mono
        elif self.channels == 2 and len(self.data.shape) == 1:
            self.data = np.column_stack([self.data, self.data])  # Convert mono to stereo

    def get_samples(self, start_frame: int, num_frames: int) -> np.ndarray:
        """Get a specific range of samples, handling looping if enabled.
        
        Parameters
        ----------
        start_frame : int
            Starting frame index
        num_frames : int
            Number of frames to retrieve
            
        Returns
        -------
        np.ndarray
            Audio samples, shape (num_frames,) for mono or (num_frames, channels) for multi-channel
        """
        if not self.loop:
            # No looping - just return the requested range (or zeros if beyond end)
            end_frame = min(start_frame + num_frames, len(self.data))
            if start_frame >= len(self.data):
                # Beyond the end of the sample
                if self.channels == 1:
                    return np.zeros(num_frames, dtype=np.float32)
                else:
                    return np.zeros((num_frames, self.channels), dtype=np.float32)
            
            # Get available data and pad with zeros if needed
            available_frames = end_frame - start_frame
            if self.channels == 1:
                result = np.zeros(num_frames, dtype=np.float32)
                result[:available_frames] = self.data[start_frame:end_frame]
            else:
                result = np.zeros((num_frames, self.channels), dtype=np.float32)
                result[:available_frames] = self.data[start_frame:end_frame]
            return result
        
        # Looping enabled - wrap around using modulo
        if self.channels == 1:
            result = np.zeros(num_frames, dtype=np.float32)
        else:
            result = np.zeros((num_frames, self.channels), dtype=np.float32)
            
        for i in range(num_frames):
            frame_idx = (start_frame + i) % len(self.data)
            if self.channels == 1:
                result[i] = self.data[frame_idx]
            else:
                result[i] = self.data[frame_idx]
                
        return result


class SampleManager:
    """Manages loading and caching of audio samples for zone-based soundscapes."""
    
    def __init__(self, target_sample_rate: int = 44100):
        """Initialize sample manager.
        
        Parameters
        ----------
        target_sample_rate : int
            Target sample rate for all loaded samples (samples will be resampled if needed)
        """
        self.target_sample_rate = target_sample_rate
        self._samples: Dict[str, AudioSample] = {}
        self._lock = threading.Lock()
    
    def load_sample(
        self,
        zone_id: str,
        file_path: Path,
        loop: bool = True,
        force_reload: bool = False
    ) -> AudioSample:
        """Load a WAV file as an audio sample.
        
        Parameters
        ----------
        zone_id : str
            Unique identifier for this sample (typically the zone ID)
        file_path : Path
            Path to the WAV file
        loop : bool
            Whether this sample should loop seamlessly
        force_reload : bool
            Whether to force reloading even if already cached
            
        Returns
        -------
        AudioSample
            Loaded audio sample
            
        Raises
        ------
        FileNotFoundError
            If the WAV file doesn't exist
        ValueError
            If the WAV file is corrupted or unsupported
        """
        with self._lock:
            if not force_reload and zone_id in self._samples:
                return self._samples[zone_id]
            
            if not file_path.exists():
                raise FileNotFoundError(f"WAV file not found: {file_path}")
            
            # Load WAV file
            try:
                with wave.open(str(file_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    
                    # Read raw audio data
                    raw_data = wav_file.readframes(frames)
                    
                    # Convert to numpy array based on sample width
                    if sample_width == 1:  # 8-bit
                        audio_data = np.frombuffer(raw_data, dtype=np.uint8)
                        audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                    elif sample_width == 2:  # 16-bit
                        audio_data = np.frombuffer(raw_data, dtype=np.int16)
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif sample_width == 4:  # 32-bit
                        audio_data = np.frombuffer(raw_data, dtype=np.int32)
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width} bytes")
                    
                    # Reshape for multi-channel audio
                    if channels > 1:
                        audio_data = audio_data.reshape(-1, channels)
                    
                    # Resample if needed (simple linear interpolation)
                    if sample_rate != self.target_sample_rate:
                        audio_data = self._resample(audio_data, sample_rate, self.target_sample_rate)
                    
                    # Create and cache the sample
                    sample = AudioSample(audio_data, self.target_sample_rate, channels, loop)
                    self._samples[zone_id] = sample
                    return sample
                    
            except Exception as e:
                raise ValueError(f"Failed to load WAV file {file_path}: {e}") from e
    
    def get_sample(self, zone_id: str) -> Optional[AudioSample]:
        """Get a cached sample by zone ID.
        
        Parameters
        ----------
        zone_id : str
            Zone identifier
            
        Returns
        -------
        AudioSample or None
            The cached sample, or None if not found
        """
        with self._lock:
            return self._samples.get(zone_id)
    
    def clear_cache(self) -> None:
        """Clear all cached samples."""
        with self._lock:
            self._samples.clear()
    
    def _resample(self, data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple linear interpolation resampling.
        
        For production use, consider using scipy.signal.resample or librosa.resample
        for better quality resampling.
        """
        if from_rate == to_rate:
            return data
        
        ratio = to_rate / from_rate
        is_multichannel = len(data.shape) > 1
        
        if is_multichannel:
            old_length, channels = data.shape
            new_length = int(old_length * ratio)
            
            # Create new time indices
            old_indices = np.arange(old_length)
            new_indices = np.linspace(0, old_length - 1, new_length)
            
            # Interpolate each channel separately
            resampled = np.zeros((new_length, channels), dtype=np.float32)
            for ch in range(channels):
                resampled[:, ch] = np.interp(new_indices, old_indices, data[:, ch])
            return resampled
        else:
            old_length = len(data)
            new_length = int(old_length * ratio)
            
            old_indices = np.arange(old_length)
            new_indices = np.linspace(0, old_length - 1, new_length)
            
            return np.interp(new_indices, old_indices, data).astype(np.float32)