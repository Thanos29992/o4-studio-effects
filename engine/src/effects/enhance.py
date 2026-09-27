"""Image Enhancement Effect — Real-ESRGAN general x4v3 (SR-as-enhancement).

Runs the model on a downscaled frame (320x180 -> 1280x720), which acts as:
- learned denoiser (webcam sensor noise)
- learned deblurrer (mild optical blur)
- learned sharpening (adds plausible high-freq detail)
- de-artifacting (JPEG compression artifacts)
...all in one pass. Output resolution matches pipeline (720p), so downstream
effects (background, auto-frame) see a cleaner frame.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from src.config import EnhanceConfig
from src.effects.base import BaseEffect
from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)

MODEL_NAME = "realesr-general-x4v3"
MODEL_DIR = Path(__file__).parent.parent / "models" / "weights"
MODEL_PATH = MODEL_DIR / "realesr-general-x4v3_180x320.xml"

# SR input resolution (downscaled from 720p: 1280/4=320, 720/4=180)
SR_IN_W = 320
SR_IN_H = 180
SR_OUT_W = 1280
SR_OUT_H = 720


class EnhanceEffect(BaseEffect):
    def __init__(self, model_manager: ModelManager, config: EnhanceConfig) -> None:
        super().__init__(model_manager)
        self.config = config
        self._enabled = config.enabled
        self._compiled_model: ov.CompiledModel | None = None
        self._infer_request: ov.InferRequest | None = None
        self._ready = False
        self._strength = config.strength

    def setup(self) -> None:
        if not MODEL_PATH.exists():
            logger.error(
                "Enhancement model missing at %s — download & convert first",
                MODEL_PATH,
            )
            self._ready = False
            return

        self._compiled_model = self.model_manager.compile_model(
            model_path=MODEL_PATH,
            model_name=MODEL_NAME,
        )
        self._infer_request = self._compiled_model.create_infer_request()
        self._ready = True

        logger.info(
            "Image enhancement loaded (320x180->720p SR, device=%s, strength=%.2f)",
            self.model_manager.preferred_device,
            self._strength,
        )

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run SR on downscaled frame, blend with original based on strength."""
        h, w = frame.shape[:2]

        # Downscale to SR input size
        sr_input = cv2.resize(frame, (SR_IN_W, SR_IN_H), interpolation=cv2.INTER_AREA)

        # Prepare tensor: NCHW RGB float32 [0,1]
        rgb = cv2.cvtColor(sr_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)

        # Inference
        result = self._infer_request.infer({0: src})
        out = result[self._compiled_model.outputs[0].any_name]

        # Post-process: NCHW RGB [0,1] -> HWC BGR uint8
        enhanced = out.squeeze().transpose(1, 2, 0)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        # Resize back to frame resolution (should already be 720p)
        if enhanced.shape[:2] != (h, w):
            enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LINEAR)

        # Blend with original based on strength (0 = off, 1 = full enhanced)
        if self._strength < 1.0:
            enhanced = cv2.addWeighted(frame, 1.0 - self._strength, enhanced, self._strength, 0)

        return enhanced

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled or not self._ready:
            return frame

        return self._enhance_frame(frame)

    def set_strength(self, strength: float) -> None:
        self._strength = np.clip(strength, 0.0, 1.0)

    def cleanup(self) -> None:
        self._infer_request = None
        self._compiled_model = None
        self._ready = False