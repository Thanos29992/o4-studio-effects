from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

gi.require_version("Gdk", "4.0")

from gi.repository import Adw, Gdk, GLib, Gtk

from src.config import AppConfig
from src.effects.auto_frame import AutoFrameEffect
from src.effects.background import BackgroundEffect
from src.pipeline.camera_detect import CameraInfo, detect_cameras

if TYPE_CHECKING:
    from src.pipeline.camera_pipeline import CameraPipeline

logger = logging.getLogger(__name__)


class SettingsWindow(Adw.PreferencesWindow):
    def __init__(self, app: Adw.Application, config: AppConfig, pipeline: CameraPipeline) -> None:
        super().__init__(application=app)
        self.set_title("Linux Studio Effects")
        self.set_default_size(480, 600)
        self.set_hide_on_close(True)

        self.config = config
        self.pipeline = pipeline
        self._cameras: list[CameraInfo] = detect_cameras()

        self._build_camera_page()
        self._build_background_page()
        self._build_autoframe_page()

    def _get_effect(self, effect_type: type) -> object | None:
        for effect in self.pipeline.get_effects():
            if isinstance(effect, effect_type):
                return effect
        return None

    def _build_camera_page(self) -> None:
        page = Adw.PreferencesPage(title="Camera", icon_name="camera-web-symbolic")
        self.add(page)

        group = Adw.PreferencesGroup(title="Camera Source")
        page.add(group)

        camera_row = Adw.ComboRow(title="Camera Device", subtitle="Select your webcam")
        camera_names = [cam.name for cam in self._cameras]
        if not camera_names:
            camera_names = ["No cameras detected"]
        camera_list = Gtk.StringList.new(camera_names)
        camera_row.set_model(camera_list)

        current_index = 0
        for idx, cam in enumerate(self._cameras):
            if cam.device_index == self.config.camera.device_index:
                current_index = idx
                break
        camera_row.set_selected(current_index)
        camera_row.connect("notify::selected", self._on_camera_select)
        group.add(camera_row)

        mirror_row = Adw.SwitchRow(title="Mirror / Selfie Mode", subtitle="Horizontally flip the camera feed")
        mirror_row.set_active(self.config.camera.mirror)
        mirror_row.connect("notify::active", self._on_mirror_toggle)
        group.add(mirror_row)

        preview_group = Adw.PreferencesGroup(title="Preview")
        page.add(preview_group)

        preview_row = Adw.SwitchRow(title="Show Preview Window", subtitle="Open a live preview of the camera with effects")
        preview_row.set_active(False)
        preview_row.connect("notify::active", self._on_preview_toggle)
        preview_group.add(preview_row)
        self._preview_active = False

    def _on_camera_select(self, row: Adw.ComboRow, _param: object) -> None:
        selected = row.get_selected()
        if selected < len(self._cameras):
            camera = self._cameras[selected]
            if camera.device_index == self.config.camera.device_index:
                return
            self.config.camera.device_index = camera.device_index
            logger.info("Camera selected: %s (device %d)", camera.name, camera.device_index)

            app = self.get_application()
            if hasattr(app, "restart_pipeline"):
                app.restart_pipeline()

    def _on_mirror_toggle(self, row: Adw.SwitchRow, _param: object) -> None:
        self.config.camera.mirror = row.get_active()
        logger.info("Mirror: %s", self.config.camera.mirror)

    def _on_preview_toggle(self, row: Adw.SwitchRow, _param: object) -> None:
        if row.get_active():
            self._preview_active = True
            app = self.get_application()
            self._preview_window = Gtk.ApplicationWindow(application=app)
            self._preview_window.set_title("Linux Studio Effects - Preview")
            self._preview_window.set_default_size(640, 480)
            self._preview_window.connect("close-request", self._on_preview_close)

            header = Gtk.HeaderBar()
            self._preview_window.set_titlebar(header)

            self._preview_picture = Gtk.Picture()
            self._preview_picture.set_content_fit(Gtk.ContentFit.CONTAIN)
            self._preview_window.set_child(self._preview_picture)

            self._preview_window.present()

            self._preview_timer_id = GLib.timeout_add(66, self._update_preview)
            logger.info("Preview window opened")
        else:
            self._stop_preview()

    def _update_preview(self) -> bool:
        if not self._preview_active:
            return False

        frame = self.pipeline.last_frame
        if frame is None:
            return True

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        stride = width * channels

        gbytes = GLib.Bytes.new(frame_rgb.tobytes())
        texture = Gdk.MemoryTexture.new(width, height, Gdk.MemoryFormat.R8G8B8, gbytes, stride)
        self._preview_picture.set_paintable(texture)

        return True

    def _on_preview_close(self, _window: Gtk.Window) -> bool:
        self._stop_preview()
        return False

    def _stop_preview(self) -> None:
        self._preview_active = False
        if hasattr(self, "_preview_timer_id"):
            GLib.source_remove(self._preview_timer_id)
        if hasattr(self, "_preview_window") and self._preview_window is not None:
            self._preview_window.close()
            self._preview_window = None
        logger.info("Preview window closed")

    def _build_background_page(self) -> None:
        page = Adw.PreferencesPage(title="Background", icon_name="camera-photo-symbolic")
        self.add(page)

        group = Adw.PreferencesGroup(title="Background Blur")
        page.add(group)

        enable_row = Adw.SwitchRow(title="Enable Background Blur", subtitle="Blur the background using AI segmentation")
        enable_row.set_active(self.config.effects.background.enabled)
        enable_row.connect("notify::active", self._on_background_toggle)
        group.add(enable_row)

        mode_row = Adw.ComboRow(title="Mode")
        mode_list = Gtk.StringList.new(["Blur", "Replace", "None"])
        mode_row.set_model(mode_list)
        mode_map = {"blur": 0, "replace": 1, "none": 2}
        mode_row.set_selected(mode_map.get(self.config.effects.background.mode, 0))
        mode_row.connect("notify::selected", self._on_background_mode)
        group.add(mode_row)

        blur_row = Adw.ActionRow(title="Blur Strength")
        blur_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 11, 101, 10)
        blur_scale.set_value(self.config.effects.background.blur_strength)
        blur_scale.set_hexpand(True)
        blur_scale.set_valign(Gtk.Align.CENTER)
        blur_scale.connect("value-changed", self._on_blur_strength)
        blur_row.add_suffix(blur_scale)
        group.add(blur_row)

    def _build_autoframe_page(self) -> None:
        page = Adw.PreferencesPage(title="Auto-Frame", icon_name="view-reveal-symbolic")
        self.add(page)

        group = Adw.PreferencesGroup(title="Auto-Framing")
        page.add(group)

        enable_row = Adw.SwitchRow(title="Enable Auto-Frame", subtitle="Automatically track and center your face")
        enable_row.set_active(self.config.effects.auto_frame.enabled)
        enable_row.connect("notify::active", self._on_autoframe_toggle)
        group.add(enable_row)

        zoom_row = Adw.ActionRow(title="Zoom Level")
        zoom_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 100.0, 1.0)
        zoom_scale.set_value(self._zoom_to_slider(self.config.effects.auto_frame.zoom_margin))
        zoom_scale.set_hexpand(True)
        zoom_scale.set_valign(Gtk.Align.CENTER)
        zoom_scale.connect("value-changed", self._on_zoom_margin)
        zoom_row.add_suffix(zoom_scale)
        group.add(zoom_row)

        smooth_row = Adw.ActionRow(title="Smoothing")
        smooth_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.01, 0.2, 0.01)
        smooth_scale.set_value(self.config.effects.auto_frame.smoothing_factor)
        smooth_scale.set_hexpand(True)
        smooth_scale.set_valign(Gtk.Align.CENTER)
        smooth_scale.connect("value-changed", self._on_smoothing)
        smooth_row.add_suffix(smooth_scale)
        group.add(smooth_row)

    def _on_background_toggle(self, row: Adw.SwitchRow, _param: object) -> None:
        effect = self._get_effect(BackgroundEffect)
        if effect is not None:
            effect.enabled = row.get_active()
            logger.info("Background effect: %s", effect.enabled)

    def _on_background_mode(self, row: Adw.ComboRow, _param: object) -> None:
        modes = ["blur", "replace", "none"]
        selected = row.get_selected()
        if selected < len(modes):
            effect = self._get_effect(BackgroundEffect)
            if isinstance(effect, BackgroundEffect):
                effect.config.mode = modes[selected]
                logger.info("Background mode: %s", modes[selected])

    def _on_blur_strength(self, scale: Gtk.Scale) -> None:
        value = int(scale.get_value()) | 1
        effect = self._get_effect(BackgroundEffect)
        if isinstance(effect, BackgroundEffect):
            effect.config.blur_strength = value

    def _on_autoframe_toggle(self, row: Adw.SwitchRow, _param: object) -> None:
        effect = self._get_effect(AutoFrameEffect)
        if effect is not None:
            effect.enabled = row.get_active()
            logger.info("Auto-frame effect: %s", effect.enabled)

    @staticmethod
    def _zoom_to_slider(margin: float) -> float:
        margin = max(1.0, min(margin, 3.0))
        return (3.0 - margin) / 2.0 * 100.0

    @staticmethod
    def _slider_to_zoom(slider: float) -> float:
        return 3.0 - (slider / 100.0) * 2.0

    def _on_zoom_margin(self, scale: Gtk.Scale) -> None:
        effect = self._get_effect(AutoFrameEffect)
        if isinstance(effect, AutoFrameEffect):
            effect.config.zoom_margin = self._slider_to_zoom(scale.get_value())

    def _on_smoothing(self, scale: Gtk.Scale) -> None:
        effect = self._get_effect(AutoFrameEffect)
        if isinstance(effect, AutoFrameEffect):
            effect.config.smoothing_factor = scale.get_value()

