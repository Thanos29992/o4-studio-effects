from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import psutil

os.environ["QT_QPA_PLATFORM"] = "xcb"

from src.config import load_config
from src.pipeline.camera_pipeline import CameraPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Studio Effects — NPU-accelerated camera effects")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to TOML config file (default: config/default.toml)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a local preview window instead of virtual camera output",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GTK4 settings panel with system tray",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run all effects for 100 frames and report per-effect latency",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Run control console only (camera off until powered on via web UI)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18080,
        help="Control console port (default: 18080)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.gui:
        from src.gui.app import run_gui

        run_gui(config)
        return

    if args.benchmark:
        _run_benchmark(config)
        return

    logger.info("Studio Effects engine")
    logger.info("Camera: device %d (%dx%d @ %d fps)", config.camera.device_index, config.camera.width, config.camera.height, config.camera.fps)
    logger.info("Inference device: %s (fallback: %s)", config.inference.device, config.inference.fallback_device)

    if args.preview:
        _run_preview(config)
        return

    # Default / --console: serve the live control console. Camera stays
    # released until power-on (web UI or POST /api/power?state=on).
    import time as _time

    from src.control_server import start as start_console

    start_console(args.port)
    logger.info("Console ready — open http://localhost:%d/", args.port)
    logger.info("Camera is OFF until powered on from the console.")
    try:
        while True:
            _time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down.")


def _run_preview(config: object) -> None:
    import cv2

    from src.config import AppConfig
    from src.effects.auto_frame import AutoFrameEffect
    from src.effects.background import BackgroundEffect
    from src.models.model_manager import ModelManager

    assert isinstance(config, AppConfig)

    model_manager = ModelManager(
        preferred_device=config.inference.device,
        fallback_device=config.inference.fallback_device,
    )

    effects = [
        BackgroundEffect(model_manager, config.effects.background),
        AutoFrameEffect(model_manager, config.effects.auto_frame),
    ]

    for effect in effects:
        effect.setup()

    capture = cv2.VideoCapture(config.camera.device_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)

    if not capture.isOpened():
        logger.error("Cannot open camera device %d", config.camera.device_index)
        sys.exit(1)

    mirror = config.camera.mirror
    logger.info("Preview mode — press 'q' quit, 'b' background, 'a' auto-frame, 'm' mirror")

    process = psutil.Process()
    frame_count = 0
    stats_timer = time.monotonic()
    process.cpu_percent()

    try:
        while True:
            frame_start = time.monotonic()

            ret, frame = capture.read()
            if not ret:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            effect_times: dict[str, float] = {}
            for effect in effects:
                if effect.enabled:
                    t0 = time.monotonic()
                    frame = effect.process(frame)
                    effect_times[effect.__class__.__name__] = (time.monotonic() - t0) * 1000

            frame_ms = (time.monotonic() - frame_start) * 1000
            frame_count += 1

            elapsed = time.monotonic() - stats_timer
            if elapsed >= 3.0:
                fps = frame_count / elapsed
                cpu_pct_total = process.cpu_percent() / (os.cpu_count() or 1)
                mem_mb = process.memory_info().rss / (1024 * 1024)

                active_effects = []
                for effect in effects:
                    name = effect.__class__.__name__.replace("Effect", "")
                    if effect.enabled:
                        active_effects.append(name)

                effects_str = ", ".join(active_effects) if active_effects else "none"
                timing_str = " | ".join(f"{k.replace('Effect', '')}:{v:.0f}ms" for k, v in effect_times.items())

                logger.info(
                    "FPS: %.1f | Frame: %.1fms | CPU: %.1f%% | RAM: %.0fMB | Effects: %s | %s",
                    fps, frame_ms, cpu_pct_total, mem_mb, effects_str, timing_str,
                )
                frame_count = 0
                stats_timer = time.monotonic()

            cv2.imshow("Linux Studio Effects - Preview", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                effects[0].enabled = not effects[0].enabled
                logger.info("Background effect: %s", effects[0].enabled)
            elif key == ord("a"):
                effects[1].enabled = not effects[1].enabled
                logger.info("Auto-frame effect: %s", effects[1].enabled)
            elif key == ord("m"):
                mirror = not mirror
                logger.info("Mirror: %s", mirror)
    except KeyboardInterrupt:
        logger.info("Stopping preview...")

    capture.release()
    cv2.destroyAllWindows()
    for effect in effects:
        effect.cleanup()
    model_manager.cleanup()


def _run_benchmark(config: object) -> None:
    import cv2

    from src.config import AppConfig
    from src.effects.auto_frame import AutoFrameEffect
    from src.effects.background import BackgroundEffect
    from src.models.model_manager import ModelManager

    assert isinstance(config, AppConfig)

    model_manager = ModelManager(
        preferred_device=config.inference.device,
        fallback_device=config.inference.fallback_device,
    )

    effects = [
        BackgroundEffect(model_manager, config.effects.background),
        AutoFrameEffect(model_manager, config.effects.auto_frame),
    ]

    for effect in effects:
        effect.setup()

    capture = cv2.VideoCapture(config.camera.device_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)

    if not capture.isOpened():
        logger.error("Cannot open camera device %d", config.camera.device_index)
        sys.exit(1)

    warmup_frames = 10
    benchmark_frames = 100

    logger.info("Warming up (%d frames)...", warmup_frames)
    for effect in effects:
        effect.enabled = True

    for _ in range(warmup_frames):
        ret, frame = capture.read()
        if ret:
            for effect in effects:
                effect.process(frame)

    effect_names = ["Background", "AutoFrame"]
    combos: list[tuple[str, list[bool]]] = [
        ("None", [False, False]),
        ("Background only", [True, False]),
        ("AutoFrame only", [False, True]),
        ("Background + AutoFrame", [True, True]),
    ]

    logger.info("Benchmarking %d frames per combo on device '%s'...", benchmark_frames, model_manager.preferred_device)
    print(f"\n{'Combo':<30} {'FPS':>6} {'Frame':>8} {'Background':>12} {'AutoFrame':>12}")
    print("-" * 70)

    for combo_name, enabled_flags in combos:
        for effect, flag in zip(effects, enabled_flags):
            effect.enabled = flag

        timings: dict[str, list[float]] = {name: [] for name in effect_names}
        frame_times: list[float] = []

        for _ in range(benchmark_frames):
            ret, frame = capture.read()
            if not ret:
                continue

            frame_start = time.monotonic()
            for effect, name in zip(effects, effect_names):
                if effect.enabled:
                    t0 = time.monotonic()
                    frame = effect.process(frame)
                    timings[name].append((time.monotonic() - t0) * 1000)
            frame_times.append((time.monotonic() - frame_start) * 1000)

        avg_frame = sum(frame_times) / len(frame_times) if frame_times else 0
        fps = 1000.0 / avg_frame if avg_frame > 0 else 0

        timing_strs: list[str] = []
        for name in effect_names:
            if timings[name]:
                avg = sum(timings[name]) / len(timings[name])
                timing_strs.append(f"{avg:>10.1f}ms")
            else:
                timing_strs.append(f"{'—':>12}")

        print(f"{combo_name:<30} {fps:>5.1f} {avg_frame:>7.1f}ms {''.join(timing_strs)}")

    print()
    capture.release()
    for effect in effects:
        effect.cleanup()
    model_manager.cleanup()


if __name__ == "__main__":
    main()
