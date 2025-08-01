import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from depth_providers import DepthAnythingV2Provider
from audio_mapper import (
    SimpleDepthToAudioMapper,
    Grid3DDepthMapper,
)
from audio_output import (
    StereoAudioOutput,
    OpenALAudioOutput,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def colorize(depth: np.ndarray) -> np.ndarray:
    """Convert a single-channel depth map to an RGB image for on-screen debugging."""
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    cmap = matplotlib.colormaps.get_cmap("Spectral")
    colored = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def run(camera_index: int, device: str | None, output_backend: str) -> None:  # noqa: D401
    """Capture frames, estimate depth and render spatial audio in real-time."""
    provider = DepthAnythingV2Provider(device=device or None)

    if output_backend == "stereo":
        mapper = SimpleDepthToAudioMapper()
        audio_out = StereoAudioOutput()
        debug_stereo = True
    else:  # "3d" (default)
        mapper = Grid3DDepthMapper()
        audio_out = OpenALAudioOutput()
        debug_stereo = False

    audio_out.start()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            depth = provider.get_depth(frame)
            sources = mapper.map(depth)
            audio_out.update_sources(sources)

            # --------------------------------------------------------------
            # Visualisation – RGB | Depth | Audio debug overlay
            # --------------------------------------------------------------
            depth_vis = colorize(depth)
            combined = np.hstack([frame, depth_vis])

            if debug_stereo:
                for azimuth, amp, _ in sources:
                    x = int((azimuth + 1) / 2 * combined.shape[1])
                    y = combined.shape[0] // 2
                    cv2.circle(combined, (x, y), radius=10, color=(0, 255, 0), thickness=-1)
            else:
                for x3d, y3d, z3d, gain, _ in sources:
                    # Map X/Y from −1..1 → screen space
                    x = int((x3d + 1) / 2 * combined.shape[1])
                    y = int((1 - y3d) / 2 * combined.shape[0])
                    cv2.circle(combined, (x, y), radius=8, color=(0, 255, 0), thickness=-1)

            cv2.imshow("RGB | Depth | Audio Debug", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        audio_out.stop()
        cap.release()
        cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# Entry-point – CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth → Spatial Audio")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu, mps)")
    parser.add_argument(
        "--output",
        choices=["3d", "stereo"],
        default="3d",
        help="Audio backend: '3d' = OpenAL (true spatial) | 'stereo' = simple stereo panning",
    )
    args = parser.parse_args()

    run(args.camera, args.device, args.output)
