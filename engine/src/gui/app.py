from __future__ import annotations

import logging
import signal
import threading
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, GLib, Gtk

from src.config import AppConfig, load_config
from src.gui.settings_window import SettingsWindow
from src.gui.tray import TrayIcon
from src.pipeline.camera_pipeline import CameraPipeline

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

APP_ID = "com.github.ericjchang.linux-studio-effects"


class StudioEffectsApp(Adw.Application):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(application_id=APP_ID)
        self.config = config
        self._pipeline: CameraPipeline | None = None
        self._pipeline_thread: threading.Thread | None = None
        self._pipeline_running = False
        self._tray: TrayIcon | None = None
        self._settings_window: SettingsWindow | None = None

    def do_activate(self) -> None:
        self._pipeline = CameraPipeline(self.config)
        self._pipeline.setup()

        self._settings_window = SettingsWindow(self, self.config, self._pipeline)
        self._settings_window.present()

        self._tray = TrayIcon(self, self._pipeline)
        self._tray.setup()

        self._start_pipeline()

        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, self._on_quit)

    def _start_pipeline(self) -> None:
        if self._pipeline_running:
            return
        self._pipeline_running = True
        self._pipeline_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self._pipeline_thread.start()
        logger.info("Pipeline started in background thread")

    def _run_pipeline(self) -> None:
        if self._pipeline is None:
            return
        self._pipeline.run()

    def _stop_pipeline(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline_running = False
        if self._pipeline_thread is not None:
            self._pipeline_thread.join(timeout=3)
        logger.info("Pipeline stopped")

    def toggle_pipeline(self) -> None:
        if self._pipeline_running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def restart_pipeline(self) -> None:
        logger.info("Restarting pipeline with new camera device %d...", self.config.camera.device_index)
        self._stop_pipeline()
        if self._pipeline is not None:
            self._pipeline.cleanup()
        self._pipeline = CameraPipeline(self.config)
        self._pipeline.setup()

        if self._settings_window is not None:
            self._settings_window.pipeline = self._pipeline

        self._start_pipeline()

    def show_settings(self) -> None:
        if self._settings_window is not None:
            self._settings_window.present()

    def _on_quit(self) -> bool:
        self._stop_pipeline()
        self.quit()
        return False

    def do_shutdown(self) -> None:
        self._stop_pipeline()
        if self._pipeline is not None:
            self._pipeline.cleanup()
        Adw.Application.do_shutdown(self)


def run_gui(config: AppConfig | None = None) -> None:
    if config is None:
        config = load_config()

    app = StudioEffectsApp(config)
    app.run(None)
