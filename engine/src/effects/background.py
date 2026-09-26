from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from src.config import BackgroundConfig
from src.effects.base import BaseEffect
from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)

MODEL_NAME = "rvm_matting"
MODEL_DIR = Path(__file__).parent.parent / "models" / "weights"
MODEL_PATH = MODEL_DIR / "rvm-dsr05.xml"

# RVM (Robust Video Matting, MobileNetV3) — static IR for 640x480 input with
# downsample_ratio baked to 0.5 (internal encoder runs at 320x240).
# Build with: python3 models/rvm/build_ir.py
SRC_HEIGHT = 480
SRC_WIDTH = 640
REC_SHAPES: tuple[tuple[int, ...], ...] = (
    (1, 16, 120, 160),
    (1, 20, 60, 80),
    (1, 40, 30, 40),
    (1, 64, 15, 20),
)


class BackgroundEffect(BaseEffect):
    def __init__(self, model_manager: ModelManager, config: BackgroundConfig) -> None:
        super().__init__(model_manager)
        self.config = config
        self._enabled = config.enabled
        self._compiled_model: ov.CompiledModel | None = None
        self._infer_request: ov.InferRequest | None = None
        self._recs: list[np.ndarray] = []
        self._background_image: np.ndarray | None = None
        self._video_capture: cv2.VideoCapture | None = None
        self.last_alpha_mask: np.ndarray | None = None
        self._prev_alpha: np.ndarray | None = None
        self._ready = False

    def setup(self) -> None:
        if not MODEL_PATH.exists():
            logger.error(
                "RVM matting model missing at %s — run: python3 models/rvm/build_ir.py",
                MODEL_PATH,
            )
            self._ready = False
            return

        self._compiled_model = self.model_manager.compile_model(
            model_path=MODEL_PATH,
            model_name=MODEL_NAME,
        )
        self._infer_request = self._compiled_model.create_infer_request()
        self._recs = [np.zeros(shape, dtype=np.float32) for shape in REC_SHAPES]
        self._ready = True

        logger.info("RVM matting model loaded (640x480 static, device=%s)",
                    self.model_manager.preferred_device)

        if self.config.background_image and Path(self.config.background_image).exists():
            self._background_image = cv2.imread(self.config.background_image)

    def _get_alpha_mask(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (SRC_WIDTH, SRC_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)

        result = self._infer_request.infer(
            {
                "src": src,
                "r1i": self._recs[0],
                "r2i": self._recs[1],
                "r3i": self._recs[2],
                "r4i": self._recs[3],
            }
        )
        # cycle recurrent states — RVM's built-in temporal memory
        self._recs = [result["r1o"], result["r2o"], result["r3o"], result["r4o"]]

        alpha = result["pha"].squeeze()

        frame_height, frame_width = frame.shape[:2]
        if alpha.shape != (frame_height, frame_width):
            alpha = cv2.resize(alpha, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

        # gentle feather + floor: kills background bleed without eating hair tips
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        alpha = np.clip((alpha - 0.05) / 0.9, 0.0, 1.0).astype(np.float32)

        # light EMA on top of RVM's recurrence (resets if resolution changes)
        if self._prev_alpha is not None and self._prev_alpha.shape == alpha.shape:
            alpha = 0.5 * alpha + 0.5 * self._prev_alpha
        self._prev_alpha = alpha

        return alpha

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled or self.config.mode == "none" or not self._ready:
            return frame

        alpha = self._get_alpha_mask(frame)
        self.last_alpha_mask = alpha
        alpha_3ch = np.stack([alpha] * 3, axis=-1)

        if self.config.mode == "blur":
            # sigma-based: same visual strength as the old ksize=strength kernels,
            # but far cheaper at high settings (no 95px kernel per frame)
            sigma = (self.config.blur_strength | 1) * 0.155 + 0.5
            blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma)
            composited = (frame * alpha_3ch + blurred * (1.0 - alpha_3ch)).astype(np.uint8)
        elif self.config.mode == "replace":
            background = self._get_background(frame.shape)
            composited = (frame * alpha_3ch + background * (1.0 - alpha_3ch)).astype(np.uint8)
        else:
            composited = frame

        return composited

    def _get_background(self, target_shape: tuple[int, ...]) -> np.ndarray:
        frame_height, frame_width = target_shape[:2]

        if self._video_capture is not None:
            ret, video_frame = self._video_capture.read()
            if not ret:
                self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, video_frame = self._video_capture.read()
            if ret:
                return cv2.resize(video_frame, (frame_width, frame_height)).astype(np.float32)

        if self._background_image is not None:
            return cv2.resize(self._background_image, (frame_width, frame_height)).astype(np.float32)

        return np.zeros((frame_height, frame_width, 3), dtype=np.float32)

    def compute_mask(self, frame: np.ndarray) -> np.ndarray:
        alpha = self._get_alpha_mask(frame)
        self.last_alpha_mask = alpha
        return alpha

    def set_background_image(self, image_path: str) -> None:
        self._close_video()
        if Path(image_path).exists():
            self._background_image = cv2.imread(image_path)
            self.config.background_image = image_path

    def set_video_background(self, video_path: str) -> None:
        self._close_video()
        self._background_image = None
        path = Path(video_path)
        if path.exists() and path.suffix.lower() in (".mp4", ".webm", ".avi", ".mkv", ".mov", ".gif"):
            self._video_capture = cv2.VideoCapture(video_path)
            if self._video_capture.isOpened():
                self.config.background_image = video_path
                logger.info("Video background set: %s", video_path)
            else:
                self._video_capture = None
                logger.warning("Could not open video: %s", video_path)

    def _close_video(self) -> None:
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

    def cleanup(self) -> None:
        self._close_video()
        self._infer_request = None
        self._compiled_model = None
        self._recs = []
        self._prev_alpha = None
        self._ready = False
