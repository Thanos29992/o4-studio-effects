from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

import openvino as ov

NPU_SNAP_LIB_PATH = "/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu"

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent
CACHE_DIR = MODELS_DIR / "cache"


class ModelManager:
    def __init__(self, preferred_device: str = "NPU", fallback_device: str = "CPU") -> None:
        self._load_npu_runtime()
        self.core = ov.Core()
        self.preferred_device = preferred_device
        self.fallback_device = fallback_device
        self._compiled_models: dict[str, ov.CompiledModel] = {}

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.core.set_property({"CACHE_DIR": str(CACHE_DIR)})

        cpu_count = os.cpu_count() or 4
        inference_threads = max(2, cpu_count // 4)
        self.core.set_property("CPU", {"INFERENCE_NUM_THREADS": inference_threads})
        logger.info("CPU inference threads limited to %d (of %d cores)", inference_threads, cpu_count)

        available_devices = self.core.available_devices
        logger.info("Available OpenVINO devices: %s", available_devices)

        if preferred_device not in available_devices:
            logger.warning(
                "Preferred device '%s' not available. Falling back to '%s'",
                preferred_device,
                fallback_device,
            )
            self.preferred_device = fallback_device

    @staticmethod
    def _load_npu_runtime() -> None:
        npu_lib_dir = Path(NPU_SNAP_LIB_PATH)
        if not npu_lib_dir.exists():
            return

        npu_libs = [
            "libze_loader.so",
            "libze_intel_npu.so",
            "libnpu_driver_compiler.so",
        ]

        try:
            for lib_name in npu_libs:
                lib_path = npu_lib_dir / lib_name
                if lib_path.exists():
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if NPU_SNAP_LIB_PATH not in ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{NPU_SNAP_LIB_PATH}:{ld_path}"

            logger.info("Loaded Level Zero runtime from Intel NPU snap")
        except OSError as exc:
            logger.debug("Could not load Level Zero runtime: %s", exc)

    def compile_model(
        self,
        model_path: str | Path,
        model_name: str,
        device: str | None = None,
        config: dict[str, str] | None = None,
        static_shape: tuple[int, ...] | None = None,
    ) -> ov.CompiledModel:
        if model_name in self._compiled_models:
            return self._compiled_models[model_name]

        target_device = device or self.preferred_device
        model_config = config or {}

        logger.info("Compiling model '%s' on device '%s'", model_name, target_device)

        model = self.core.read_model(str(model_path))

        if static_shape is not None:
            model.reshape(static_shape)
            logger.info("Reshaped model '%s' to static shape %s", model_name, static_shape)

        try:
            compiled = self.core.compile_model(model, target_device, model_config)
        except RuntimeError:
            logger.warning(
                "Failed to compile '%s' on '%s', falling back to '%s'",
                model_name,
                target_device,
                self.fallback_device,
            )
            compiled = self.core.compile_model(model, self.fallback_device, model_config)

        self._compiled_models[model_name] = compiled
        return compiled

    def get_model(self, model_name: str) -> ov.CompiledModel | None:
        return self._compiled_models.get(model_name)

    def cleanup(self) -> None:
        self._compiled_models.clear()
