from __future__ import annotations

"""Depth Anything V2 provider that relies on HuggingFace `transformers` instead of the
original repo.  This avoids having to install the (non-packaged) upstream code and
works out-of-the-box with `pip install transformers`.

The provider is intentionally lightweight: it loads one of the official
`*-hf` checkpoints published on the HuggingFace Hub and exposes a simple
`get_depth(frame)` API that returns a single-channel float32 depth map whose
bigger values correspond to *closer* objects.
"""

from typing import Final

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .base import DepthProvider


_DEFAULT_MODEL_ID: Final[str] = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthAnythingV2Provider(DepthProvider):
    """Depth Anything V2 wrapper that matches the simple `DepthProvider` interface.

    Parameters
    ----------
    model_id:
        Any of the `depth-anything/Depth-Anything-V2-*-hf` checkpoints on the Hub.
        Defaults to the *Small* variant which easily fits into GPU/Apple-Silicon
        memory while remaining real-time.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).  If *None* we
        automatically choose ``"mps"`` on Apple Silicon if available, otherwise
        CPU.
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        # ---- load HF model
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:  # noqa: D401
        return f"Depth Anything V2 ({self.model_id.split('/')[-1]})"

    @torch.no_grad()
    def get_depth(self, frame: np.ndarray) -> np.ndarray:  # noqa: D401
        """Estimate depth for a single *BGR* image coming from OpenCV."""
        # Convert BGR (OpenCV) → RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        pred = outputs.predicted_depth

        # Resize to original resolution
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],  # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = depth.cpu().numpy().astype(np.float32)
        return depth
