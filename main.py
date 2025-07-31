import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from depth_providers import DepthAnythingV2Provider
from audio_mapper import SimpleDepthToAudioMapper
from audio_output import StereoAudioOutput


def colorize(depth: np.ndarray) -> np.ndarray:
    """Convert a single-channel depth map to a colour image."""
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    colored = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


def run(camera_index: int, device: str | None):
    provider = DepthAnythingV2Provider(device=device or None)
    mapper = SimpleDepthToAudioMapper()
    audio_out = StereoAudioOutput()
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

            # Visualisation
            depth_vis = colorize(depth)
            combined = np.hstack([frame, depth_vis])
            # Draw debug markers where sound sources are placed
            for azimuth, amp, _ in sources:
                x = int((azimuth + 1) / 2 * combined.shape[1])
                y = combined.shape[0] // 2
                cv2.circle(combined, (x, y), radius=10, color=(0, 255, 0), thickness=-1)

            cv2.imshow("RGB | Depth | Audio Debug", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        audio_out.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth → Spatial Audio (Mac)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu, mps)")
    args = parser.parse_args()

    run(args.camera, args.device)
