from __future__ import annotations

import logging
import signal
import time

import cv2
import numpy as np
import pyvirtualcam

from src.config import AppConfig
from src.effects.auto_frame import AutoFrameEffect
from src.effects.background import BackgroundEffect
from src.effects.base import BaseEffect
from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)


class CameraPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._running = False
        self._capture: cv2.VideoCapture | None = None
        self._model_manager = ModelManager(
            preferred_device=config.inference.device,
            fallback_device=config.inference.fallback_device,
        )
        self._effects: list[BaseEffect] = []
        self._last_frame: np.ndarray | None = None
        self._last_raw: np.ndarray | None = None
        self._fps: float = 0.0

    @property
    def camera_available(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def setup(self) -> None:
        logger.info("Setting up camera pipeline...")

        self._capture = self._try_open_camera()

        if self._capture is not None:
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
            logger.info("Camera opened: %dx%d @ %.1f fps", actual_width, actual_height, actual_fps)
        else:
            logger.warning("No camera available — effects initialized but pipeline idle. Select a camera to start.")

        background_effect = BackgroundEffect(self._model_manager, self.config.effects.background)
        auto_frame_effect = AutoFrameEffect(self._model_manager, self.config.effects.auto_frame)

        self._effects = [background_effect, auto_frame_effect]

        for effect in self._effects:
            effect.setup()

        logger.info("All effects initialized")

    def _try_open_camera(self) -> cv2.VideoCapture | None:
        preferred = self.config.camera.device_index
        cap = cv2.VideoCapture(preferred)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)

        if cap.isOpened():
            return cap
        cap.release()

        logger.warning("Camera device %d busy/unavailable, trying alternatives...", preferred)

        from src.pipeline.camera_detect import detect_cameras

        for camera in detect_cameras():
            if camera.device_index == preferred:
                continue
            alt_cap = cv2.VideoCapture(camera.device_index)
            alt_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
            alt_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
            alt_cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)

            if alt_cap.isOpened():
                self.config.camera.device_index = camera.device_index
                logger.info("Switched to fallback camera: %s", camera.name)
                return alt_cap
            alt_cap.release()

        logger.warning("No cameras available")
        return None

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        self._last_raw = frame
        processed = frame
        for effect in self._effects:
            if effect.enabled:
                processed = effect.process(processed)
        self._last_frame = processed
        return processed

    @property
    def last_frame(self) -> np.ndarray | None:
        return self._last_frame

    @property
    def last_raw(self) -> np.ndarray | None:
        return self._last_raw

    @property
    def fps(self) -> float:
        return self._fps

    def run(self) -> None:
        if self._capture is None or not self._capture.isOpened():
            logger.info("No camera — pipeline idle. Waiting for camera selection.")
            self._running = True
            while self._running:
                time.sleep(0.5)
            return

        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._running = True

        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            pass

        virtual_cam = None
        try:
            import pathlib

            if pathlib.Path(self.config.virtual_camera.device_path).exists():
                logger.info(
                    "Starting virtual camera on %s (%dx%d)",
                    self.config.virtual_camera.device_path,
                    actual_width,
                    actual_height,
                )
                virtual_cam = pyvirtualcam.Camera(
                    width=actual_width,
                    height=actual_height,
                    fps=self.config.camera.fps,
                    device=self.config.virtual_camera.device_path,
                    fmt=pyvirtualcam.PixelFormat.BGR,
                )
                logger.info("Virtual camera started: %s", virtual_cam.device)
            else:
                logger.info("Virtual camera device not found — running without output (GUI-only mode)")
        except RuntimeError as exc:
            logger.warning("Could not start virtual camera: %s", exc)

        frame_count = 0
        fps_timer = time.monotonic()

        try:
            while self._running:
                ret, frame = self._capture.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue

                if self.config.camera.mirror:
                    frame = cv2.flip(frame, 1)

                processed = self._process_frame(frame)
                if virtual_cam is not None:
                    virtual_cam.send(processed)
                    virtual_cam.sleep_until_next_frame()
                else:
                    time.sleep(1.0 / self.config.camera.fps)

                frame_count += 1
                elapsed = time.monotonic() - fps_timer
                if elapsed >= 5.0:
                    actual_fps = frame_count / elapsed
                    self._fps = actual_fps
                    logger.info("Pipeline FPS: %.1f", actual_fps)
                    frame_count = 0
                    fps_timer = time.monotonic()
        finally:
            if virtual_cam is not None:
                virtual_cam.close()

        self.cleanup()

    def _signal_handler(self, signum: int, _frame: object) -> None:
        logger.info("Received signal %d, stopping pipeline...", signum)
        self._running = False

    def stop(self) -> None:
        self._running = False

    def cleanup(self) -> None:
        logger.info("Cleaning up pipeline...")
        for effect in self._effects:
            effect.cleanup()
        self._model_manager.cleanup()
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def get_effects(self) -> list[BaseEffect]:
        return self._effects
