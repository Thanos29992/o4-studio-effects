"""Live control console for Studio Effects.

Single HTTP server (default :18080) that provides:
  GET  /                     -> camera.html live console
  GET  /api/status           -> JSON: power, effects, fps, npu, cpu, ram
  POST /api/power?state=on|off   -> open/release camera + start/stop pipeline
  POST /api/effect?name=..&state=on|off -> toggle effect live (+ sync state file)
  POST /api/blur?value=0..100     -> blur strength (syncs state/blur_strength)
  POST /api/autoframe?smoothing=..&zoom=..&interval=..&confidence=..&headroom=..&transition=..
  GET  /stream/raw.mjpg      -> MJPEG of unprocessed camera
  GET  /stream/processed.mjpg-> MJPEG of processed output

Runs inside the engine process (thread). Power off fully releases /dev/video*.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent  # .../studio-effects
STATE_DIR = ROOT / "state"
CONSOLE_HTML = ROOT / "tools" / "test-page" / "camera.html"

NPU_BUSY_PATH = Path("/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us")

# web effect name -> state file suffix
EFFECT_MAP = {
    "background": "modnet",
    "auto_frame": "face-adas",
}

# auto-framing tunables: query key -> (config attr, caster, min, max, state file)
AF_PARAMS = {
    "smoothing":  ("smoothing_factor",     float, 0.01, 0.5,  "af_smoothing"),
    "zoom":       ("zoom_margin",          float, 1.0,  3.0,  "af_zoom"),
    "interval":   ("detection_interval",   int,   1,    15,   "af_interval"),
    "confidence": ("confidence_threshold", float, 0.1,  0.9,  "af_confidence"),
    "headroom":   ("headroom",             float, 0.0,  0.5,  "af_headroom"),
    "transition": ("transition_speed",     float, 0.01, 0.2,  "af_transition"),
}

# JS/CSS assets are inlined in camera.html; nothing else to serve.


def _read_state(name: str, default: str = "") -> str:
    try:
        return (STATE_DIR / name).read_text().strip()
    except OSError:
        return default


def _write_state(name: str, value: str) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / name).write_text(value + "\n")
    except OSError as exc:
        logger.warning("Could not write state %s: %s", name, exc)


class EngineController:
    """Owns the CameraPipeline lifecycle: power on/off + live effect toggles."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipeline = None
        self._thread: threading.Thread | None = None
        self._power = False
        self._npu_sample: tuple[float, int] | None = None

    # ── power ────────────────────────────────────────────────────────────

    @property
    def power(self) -> bool:
        return self._power

    def power_on(self) -> dict:
        with self._lock:
            if self._power:
                return {"ok": True, "power": True, "note": "already on"}
            from src.config import load_config
            from src.pipeline.camera_pipeline import CameraPipeline

            config = load_config()
            self._apply_state_files(config)
            pipeline = CameraPipeline(config)
            pipeline.setup()
            self._pipeline = pipeline
            self._thread = threading.Thread(target=pipeline.run, daemon=True, name="pipeline")
            self._thread.start()
            self._power = True
            logger.info("Power ON — camera pipeline started")
            return {"ok": True, "power": True}

    def power_off(self) -> dict:
        with self._lock:
            if not self._power:
                return {"ok": True, "power": False, "note": "already off"}
            pipeline = self._pipeline
            if pipeline is not None:
                pipeline.stop()
            thread = self._thread
            self._pipeline = None
            self._thread = None
            self._power = False
        if thread is not None:
            thread.join(timeout=5.0)
        logger.info("Power OFF — camera released")
        return {"ok": True, "power": False}

    # ── state-file sync (the QML panel contract) ─────────────────────────

    def _apply_state_files(self, config) -> None:
        """Seed effect enable flags + blur strength from state/ files."""
        def on(suffix: str) -> bool:
            return _read_state(f"effect_{suffix}", "off") == "on"

        config.effects.background.enabled = on("modnet")
        config.effects.auto_frame.enabled = on("face-adas")

        try:
            config.effects.background.blur_strength = int(_read_state("blur_strength", "50"))
        except ValueError:
            pass

        mode = _read_state("bg_mode", "")
        if mode in ("blur", "replace", "none"):
            config.effects.background.mode = mode

        bg_image = _read_state("bg_image_path", "")
        if bg_image:
            config.effects.background.background_image = bg_image

        # auto-framing tunables (live-tuned values persist across power cycles)
        af = config.effects.auto_frame
        for attr, caster, lo, hi, state_name in AF_PARAMS.values():
            raw = _read_state(state_name, "")
            if raw:
                try:
                    setattr(af, attr, max(lo, min(hi, caster(raw))))
                except ValueError:
                    pass

    def set_effect(self, name: str, enabled: bool) -> dict:
        suffix = EFFECT_MAP.get(name, "__invalid__")
        if suffix == "__invalid__":
            return {"ok": False, "error": f"unknown effect '{name}'"}
        if suffix:
            _write_state(f"effect_{suffix}", "on" if enabled else "off")

        with self._lock:
            pipeline = self._pipeline
            if pipeline is None:
                return {"ok": True, "effect": name, "enabled": enabled, "power": False,
                        "note": "state saved; applies on next power-on"}
            for effect in pipeline.get_effects():
                key = type(effect).__name__
                if (name == "background" and key == "BackgroundEffect") or \
                   (name == "auto_frame" and key == "AutoFrameEffect"):
                    effect.enabled = enabled
                    return {"ok": True, "effect": name, "enabled": enabled, "power": True}
        return {"ok": False, "error": "effect not found in pipeline"}

    def set_autoframe(self, params: dict[str, str]) -> dict:
        """Update auto-framing tunables (any subset of AF_PARAMS keys).

        Clamps, writes state files (survive power cycles), and applies live
        to the running effect so changes take effect on the next frame.
        """
        applied: dict[str, float | int] = {}
        unknown = [k for k in params if k not in AF_PARAMS]
        for key, raw in params.items():
            spec = AF_PARAMS.get(key)
            if spec is None:
                continue
            attr, caster, lo, hi, state_name = spec
            try:
                value = caster(float(raw))
            except ValueError:
                continue
            if caster is int:
                value = int(value)
            value = max(lo, min(hi, value))
            _write_state(state_name, str(value))
            applied[key] = value
            with self._lock:
                pipeline = self._pipeline
                if pipeline is not None:
                    for effect in pipeline.get_effects():
                        if type(effect).__name__ == "AutoFrameEffect":
                            setattr(effect.config, attr, value)
        out: dict = {"ok": bool(applied), "applied": applied}
        if unknown:
            out["unknown"] = unknown
        return out

    def set_blur(self, value: int) -> dict:
        value = max(1, min(99, value))
        _write_state("blur_strength", str(value))
        with self._lock:
            pipeline = self._pipeline
            if pipeline is not None:
                for effect in pipeline.get_effects():
                    if type(effect).__name__ == "BackgroundEffect":
                        effect.config.blur_strength = value | 1
        return {"ok": True, "blur": value}

    def set_mode(self, mode: str) -> dict:
        """Background mode: blur | replace | none (background changing)."""
        if mode not in ("blur", "replace", "none"):
            return {"ok": False, "error": "mode must be blur|replace|none"}
        _write_state("bg_mode", mode)
        with self._lock:
            pipeline = self._pipeline
            if pipeline is not None:
                for effect in pipeline.get_effects():
                    if type(effect).__name__ == "BackgroundEffect":
                        effect.config.mode = mode
                        # changing mode turns the whole background effect on
                        effect.enabled = mode != "none"
        if mode != "none":
            _write_state("effect_modnet", "on")
        return {"ok": True, "mode": mode}

    def set_bg_image(self, path: str) -> dict:
        """Set background replacement image (absolute path)."""
        from pathlib import Path as _P
        p = _P(path).expanduser()
        if not p.is_file():
            return {"ok": False, "error": f"file not found: {path}"}
        _write_state("bg_image_path", str(p))
        with self._lock:
            pipeline = self._pipeline
            if pipeline is not None:
                for effect in pipeline.get_effects():
                    if type(effect).__name__ == "BackgroundEffect":
                        effect.set_background_image(str(p))
        return {"ok": True, "bg_image": str(p)}

    # ── status ───────────────────────────────────────────────────────────

    def _npu_percent(self) -> float:
        try:
            busy = int(NPU_BUSY_PATH.read_text())
        except (OSError, ValueError):
            return -1.0
        now = time.monotonic()
        if self._npu_sample is None:
            self._npu_sample = (now, busy)
            return 0.0
        prev_t, prev_b = self._npu_sample
        dt = now - prev_t
        if dt < 0.3:
            return 0.0
        self._npu_sample = (now, busy)
        return round(min(100.0, (busy - prev_b) / 1e6 / dt * 100.0), 1)

    def status(self) -> dict:
        import os
        import psutil

        with self._lock:
            pipeline = self._pipeline
            power = self._power
            effects_state = {}
            fps = None
            mode = None
            af_live = None
            if pipeline is not None:
                for effect in pipeline.get_effects():
                    key = {"BackgroundEffect": "background",
                           "AutoFrameEffect": "auto_frame"}.get(type(effect).__name__)
                    if key:
                        effects_state[key] = bool(effect.enabled)
                    if type(effect).__name__ == "BackgroundEffect":
                        mode = effect.config.mode
                    if type(effect).__name__ == "AutoFrameEffect":
                        af_live = {
                            web_key: getattr(effect.config, attr)
                            for web_key, (attr, *_rest) in AF_PARAMS.items()
                        }
                fps = getattr(pipeline, "fps", None)

        if af_live is None:
            # powered off: report state-file values (fall back to config defaults)
            from src.config import load_config
            af_cfg = load_config().effects.auto_frame
            af_live = {
                web_key: getattr(af_cfg, attr)
                for web_key, (attr, *_rest) in AF_PARAMS.items()
            }

        proc = psutil.Process()
        return {
            "power": power,
            "effects": effects_state,
            "blur": int(_read_state("blur_strength", "50") or 50),
            "mode": mode or _read_state("bg_mode", "blur"),
            "autoframe": af_live,
            "fps": round(fps, 1) if fps else None,
            "npu_percent": self._npu_percent(),
            "cpu_percent": round(proc.cpu_percent(interval=0.0) / (os.cpu_count() or 1), 1),
            "ram_mb": round(proc.memory_info().rss / (1024 * 1024)),
            "camera": power,
        }

    # ── frames for MJPEG ─────────────────────────────────────────────────

    def frames(self, kind: str):
        """Yield (raw, processed) BGR frames while powered; None when off."""
        while True:
            with self._lock:
                pipeline = self._pipeline
                power = self._power
            if not power or pipeline is None:
                yield None, None
                time.sleep(0.25)
                continue
            raw = getattr(pipeline, "last_raw", None)
            processed = pipeline.last_frame
            yield raw, processed
            time.sleep(1.0 / 30.0)


_controller = EngineController()


def _mjpeg_response(handler: BaseHTTPRequestHandler, kind: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    try:
        for raw, processed in _controller.frames(kind):
            frame = raw if kind == "raw" else processed
            if frame is None:
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
            handler.wfile.write(buf.tobytes())
            handler.wfile.write(b"\r\n")
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # silence per-request logs
        pass

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _query(self) -> dict:
        return parse_qs(urlparse(self.path).query)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html", "/camera.html"):
            try:
                body = CONSOLE_HTML.read_bytes()
            except OSError:
                self._json({"error": "console page missing"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self._json(_controller.status())
        elif path == "/stream/raw.mjpg":
            _mjpeg_response(self, "raw")
        elif path == "/stream/processed.mjpg":
            _mjpeg_response(self, "processed")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        q = {k: v[0] for k, v in self._query().items()}
        if parsed.path == "/api/power":
            state = q.get("state", "")
            if state == "on":
                self._json(_controller.power_on())
            elif state == "off":
                self._json(_controller.power_off())
            else:
                self._json({"ok": False, "error": "state must be on|off"}, 400)
        elif parsed.path == "/api/effect":
            name = q.get("name", "")
            state = q.get("state", "")
            if not name or state not in ("on", "off"):
                self._json({"ok": False, "error": "need name + state=on|off"}, 400)
                return
            self._json(_controller.set_effect(name, state == "on"))
        elif parsed.path == "/api/autoframe":
            self._json(_controller.set_autoframe(q))
        elif parsed.path == "/api/blur":
            try:
                value = int(q.get("value", ""))
            except ValueError:
                self._json({"ok": False, "error": "need value=0..100"}, 400)
                return
            self._json(_controller.set_blur(value))
        elif parsed.path == "/api/mode":
            self._json(_controller.set_mode(q.get("value", "")))
        elif parsed.path == "/api/bgimage":
            self._json(_controller.set_bg_image(q.get("path", "")))
        else:
            self._json({"error": "not found"}, 404)


def start(port: int = 18080) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="control-http").start()
    logger.info("Control console: http://localhost:%d/", port)
    return server
