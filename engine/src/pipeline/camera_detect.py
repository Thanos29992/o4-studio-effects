from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    device_index: int
    name: str
    is_color: bool


def detect_cameras() -> list[CameraInfo]:
    cameras: list[CameraInfo] = []

    by_id = Path("/dev/v4l/by-id")
    if not by_id.exists():
        return cameras

    for link in sorted(by_id.iterdir()):
        link_name = link.name

        if "v4l2loopback" in link_name:
            continue

        if "video-index0" not in link_name:
            continue

        target = link.resolve().name
        if not target.startswith("video"):
            continue

        try:
            device_index = int(target.replace("video", ""))
        except ValueError:
            continue

        name = link_name.replace("usb-", "").split("-video-")[0].replace("_", " ").strip()
        display_name = f"{name} [/dev/video{device_index}]"

        cameras.append(CameraInfo(
            device_index=device_index,
            name=display_name,
            is_color=True,
        ))

    logger.info("Detected %d camera(s): %s", len(cameras), [c.name for c in cameras])
    return cameras
