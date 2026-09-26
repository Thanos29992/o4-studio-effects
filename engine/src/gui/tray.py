from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.gui.app import StudioEffectsApp
    from src.pipeline.camera_pipeline import CameraPipeline

logger = logging.getLogger(__name__)


class TrayIcon:
    """System tray via a separate GTK3 subprocess (avoids GTK3/4 conflict)."""

    def __init__(self, app: StudioEffectsApp, pipeline: CameraPipeline) -> None:
        self.app = app
        self.pipeline = pipeline
        self._process: subprocess.Popen[bytes] | None = None

    def setup(self) -> None:
        try:
            tray_script = str(__import__("pathlib").Path(__file__).parent / "tray_subprocess.py")
            self._process = subprocess.Popen(
                ["/usr/bin/python3", tray_script],
            )
            logger.info("System tray started (subprocess PID: %d)", self._process.pid)
        except Exception as exc:
            logger.warning("Could not start system tray: %s", exc)

    def cleanup(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process = None
