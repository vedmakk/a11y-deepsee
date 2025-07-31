# Depth Spatial Audio

Hear your surroundings with computer vision. This _experimental_ Python application captures live video from the built-in camera on macOS (Apple Silicon), estimates depth with Depth Anything V2 and converts the depth map into spatialised stereo audio in real-time.

![Screenshot](./assets/screenshot.png)

## Features

- Live camera feed → depth map using state-of-the-art Depth Anything V2 (runs on Apple M-series GPU).
- Spatial audio where nearer objects sound louder and their horizontal position is perceived via stereo panning.
- Modular architecture:
  - `DepthProvider` – swap the depth model.
  - `DepthToAudioMapper` – implement your own mapping from depth to audio sources.
  - `AudioOutput` – replace the audio back-end or use more sophisticated HRTF renderers.
- Simple OpenCV UI that shows:
  - raw RGB feed
  - colour-coded depth map
  - green dot that indicates where the audio source is panned
- Runs locally (no internet connection once the model checkpoint has been downloaded the first time).

> NOTE This proof-of-concept uses simple stereo panning – it does **not** use Apple’s head-tracked spatial audio APIs. You can improve fidelity by implementing an `AudioOutput` that uses a true binaural renderer (e.g. `pyroomacoustics`).

## Installation

1. Install Python 3.10 + (easiest via [Homebrew](https://brew.sh/))

   ```bash
   brew install python
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The first run will download the Depth Anything V2 checkpoint (~250 MB) from the HuggingFace Hub.

## Usage

```bash
python main.py               # press “q” to quit
```

Optional flags:

- `--camera 1` use external camera with index 1
- `--device cpu` force CPU inference (slow)

## Customising the Mapping

`audio_mapper/simple_mapper.py` exposes several knobs:

- `grid_size` resolution of the grid that is sampled
- `min_depth` / `max_depth` depth range in metres that should produce sound
- `base_freq` / `freq_span` frequency range (Hz)

Feel free to subclass `DepthToAudioMapper` to implement alternative sonifications (e.g. different waveforms, MIDI output, per-pixel granular synthesis, etc.).

## Project Structure

```
depth_spatial_audio/
├── audio_mapper/
│   ├── base.py
│   ├── simple_mapper.py
├── audio_output/
│   ├── base.py
│   ├── stereo_output.py
├── depth_providers/
│   ├── base.py
│   ├── depth_anything_v2.py
├── main.py
├── requirements.txt
└── README.md
```

## Requirements & Tested Environment

- macOS 15.5 Sequoia on **Apple M1**
- Python 3.11
- PyTorch 2.3.0 (Metal / MPS backend)
- OpenCV 4.10.0
- sounddevice 0.4.6

Other Apple Silicon models should work. Intel Macs are untested. Windows / Linux users will need to swap the audio backend and maybe adjust the camera index.

---

Have fun exploring your surroundings _by ear_! PRs improving the depth-to-audio mapping or adding proper HRTF support are very welcome.
