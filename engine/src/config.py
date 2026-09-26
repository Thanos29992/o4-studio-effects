from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.toml"


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True


@dataclass
class VirtualCameraConfig:
    device_path: str = "/dev/video10"
    label: str = "StudioEffects"


@dataclass
class InferenceConfig:
    device: str = "NPU"
    fallback_device: str = "CPU"


@dataclass
class BackgroundConfig:
    enabled: bool = True
    mode: Literal["blur", "replace", "none"] = "blur"
    blur_strength: int = 51
    background_image: str = ""


@dataclass
class AutoFrameConfig:
    enabled: bool = False
    zoom_margin: float = 1.3
    smoothing_factor: float = 0.15
    detection_interval: float = 1.0  # seconds between face detections (CPU poll rate)
    confidence_threshold: float = 0.5
    headroom: float = 0.18
    transition_speed: float = 0.05
    deadzone: float = 0.08  # face center within this radius of held pose -> hold (no recenter)
    size_deadzone: float = 0.15  # relative face-size change below this -> hold (no zoom)


@dataclass
class EffectsConfig:
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    auto_frame: AutoFrameConfig = field(default_factory=AutoFrameConfig)


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    virtual_camera: VirtualCameraConfig = field(default_factory=VirtualCameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    effects: EffectsConfig = field(default_factory=EffectsConfig)


def load_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return AppConfig()

    with open(path, "rb") as config_file:
        raw = tomllib.load(config_file)

    camera_raw = raw.get("camera", {})
    virtual_camera_raw = raw.get("virtual_camera", {})
    inference_raw = raw.get("inference", {})
    effects_raw = raw.get("effects", {})

    return AppConfig(
        camera=CameraConfig(**camera_raw),
        virtual_camera=VirtualCameraConfig(**virtual_camera_raw),
        inference=InferenceConfig(**inference_raw),
        effects=EffectsConfig(
            background=BackgroundConfig(**effects_raw.get("background", {})),
            auto_frame=AutoFrameConfig(**effects_raw.get("auto_frame", {})),
        ),
    )
