from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from src.config import AutoFrameConfig
from src.effects.base import BaseEffect
from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)

FACE_DET_MODEL_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/"
    "face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
)
FACE_DET_WEIGHTS_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/"
    "face-detection-adas-0001/FP16/face-detection-adas-0001.bin"
)

MODEL_NAME = "face_detection"
MODEL_DIR = Path(__file__).parent.parent / "models" / "weights"
MODEL_XML = MODEL_DIR / "face-detection-adas-0001.xml"
MODEL_BIN = MODEL_DIR / "face-detection-adas-0001.bin"

CONFIDENCE_THRESHOLD = 0.5


class AutoFrameEffect(BaseEffect):
    def __init__(self, model_manager: ModelManager, config: AutoFrameConfig) -> None:
        super().__init__(model_manager)
        self.config = config
        self._enabled = config.enabled
        self._compiled_model: ov.CompiledModel | None = None
        self._infer_request: ov.InferRequest | None = None
        self._input_shape: tuple[int, ...] = ()
        self._frame_count = 0
        self._smooth_center_x = 0.5
        self._smooth_center_y = 0.5
        self._smooth_face_size = 0.15
        self._initialized = False
        self._zoom_level = 1.0 if config.enabled else 0.0
        self._transitioning = False

    def setup(self) -> None:
        self._download_model_if_needed()

        self._compiled_model = self.model_manager.compile_model(
            model_path=MODEL_XML,
            model_name=MODEL_NAME,
            device="CPU",
        )
        self._infer_request = self._compiled_model.create_infer_request()

        input_layer = self._compiled_model.input(0)
        self._input_shape = tuple(input_layer.shape)
        logger.info("Face detection model loaded. Input shape: %s", self._input_shape)

    def _download_model_if_needed(self) -> None:
        if MODEL_XML.exists() and MODEL_BIN.exists():
            return

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading face detection model...")

        import urllib.request

        urllib.request.urlretrieve(FACE_DET_MODEL_URL, MODEL_XML)
        urllib.request.urlretrieve(FACE_DET_WEIGHTS_URL, MODEL_BIN)
        logger.info("Model downloaded to %s", MODEL_DIR)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        _, _, input_height, input_width = self._input_shape
        resized = cv2.resize(frame, (input_width, input_height))
        transposed = resized.transpose(2, 0, 1)
        return np.expand_dims(transposed.astype(np.float32), axis=0)

    def _detect_faces(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        input_tensor = self._preprocess(frame)
        result = self._infer_request.infer({0: input_tensor})
        output_key = list(result.keys())[0]
        detections = result[output_key].squeeze()

        faces: list[tuple[float, float, float, float]] = []
        for detection in detections:
            confidence = detection[2]
            if confidence > CONFIDENCE_THRESHOLD:
                x_min = float(np.clip(detection[3], 0.0, 1.0))
                y_min = float(np.clip(detection[4], 0.0, 1.0))
                x_max = float(np.clip(detection[5], 0.0, 1.0))
                y_max = float(np.clip(detection[6], 0.0, 1.0))
                faces.append((x_min, y_min, x_max, y_max))

        return faces

    def _update_smooth_target(self, faces: list[tuple[float, float, float, float]]) -> None:
        if not faces:
            return

        if len(faces) == 1:
            x_min, y_min, x_max, y_max = faces[0]
        else:
            x_min = min(f[0] for f in faces)
            y_min = min(f[1] for f in faces)
            x_max = max(f[2] for f in faces)
            y_max = max(f[3] for f in faces)

        target_cx = (x_min + x_max) / 2.0
        target_cy = (y_min + y_max) / 2.0
        target_size = max(x_max - x_min, y_max - y_min)

        if not self._initialized:
            self._smooth_center_x = target_cx
            self._smooth_center_y = target_cy
            self._smooth_face_size = target_size
            self._initialized = True
            logger.info(
                "Face detected: center=(%.2f, %.2f) size=%.3f",
                target_cx, target_cy, target_size,
            )
            return

        alpha = self.config.smoothing_factor
        self._smooth_center_x += alpha * (target_cx - self._smooth_center_x)
        self._smooth_center_y += alpha * (target_cy - self._smooth_center_y)
        self._smooth_face_size += alpha * 0.5 * (target_size - self._smooth_face_size)

    @BaseEffect.enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        self._transitioning = True

    def process(self, frame: np.ndarray) -> np.ndarray:
        target_zoom = 1.0 if self._enabled else 0.0

        if self._transitioning:
            self._zoom_level += 0.05 * (target_zoom - self._zoom_level)
            if abs(self._zoom_level - target_zoom) < 0.01:
                self._zoom_level = target_zoom
                self._transitioning = False

        if self._zoom_level < 0.01:
            return frame

        if self._enabled:
            self._frame_count += 1
            if self._frame_count % self.config.detection_interval == 0:
                faces = self._detect_faces(frame)
                self._update_smooth_target(faces)

        return self._apply_crop(frame)

    def _apply_crop(self, frame: np.ndarray) -> np.ndarray:
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height

        face_height_px = self._smooth_face_size * frame_height
        face_center_y = self._smooth_center_y * frame_height
        face_center_x = self._smooth_center_x * frame_width

        face_top = face_center_y - face_height_px * 0.65
        face_bottom = face_center_y + face_height_px * 0.5

        target_crop_h = face_height_px * self.config.zoom_margin
        min_crop_for_face = (face_bottom - face_top) * 1.3
        target_crop_h = max(target_crop_h, min_crop_for_face)
        target_crop_h = max(target_crop_h, frame_height * 0.25)
        target_crop_h = min(target_crop_h, float(frame_height))

        crop_h_px = frame_height + self._zoom_level * (target_crop_h - frame_height)
        crop_w_px = crop_h_px * aspect_ratio
        crop_w_px = min(crop_w_px, float(frame_width))
        crop_h_px = crop_w_px / aspect_ratio

        ideal_crop_top = face_top - crop_h_px * 0.18
        target_cy = ideal_crop_top + crop_h_px * 0.5
        target_cx = face_center_x

        center_x_px = frame_width / 2 + self._zoom_level * (target_cx - frame_width / 2)
        center_y_px = frame_height / 2 + self._zoom_level * (target_cy - frame_height / 2)

        x1 = int(np.clip(center_x_px - crop_w_px / 2, 0, frame_width - crop_w_px))
        y1 = int(np.clip(center_y_px - crop_h_px / 2, 0, frame_height - crop_h_px))

        if y1 > face_top - face_height_px * 0.2:
            y1 = int(max(0, face_top - face_height_px * 0.2))
        if y1 + crop_h_px < face_bottom + face_height_px * 0.1:
            y1 = int(max(0, face_bottom + face_height_px * 0.1 - crop_h_px))

        x2 = int(min(x1 + crop_w_px, frame_width))
        y2 = int(min(y1 + crop_h_px, frame_height))

        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            return frame

        return cv2.resize(cropped, (frame_width, frame_height))

    def cleanup(self) -> None:
        self._infer_request = None
        self._compiled_model = None
