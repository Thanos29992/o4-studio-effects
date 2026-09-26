"""Build the RVM NPU IR used by BackgroundEffect.

Pipeline (each step exists because of a specific failure):
  1. Download rvm_mobilenetv3_fp32.onnx (SourceForge v1.0.0 mirror) if missing.
  2. Upgrade opset 12 -> 17 (original was pytorch 1.9 / ir_version 6).
  3. Bake `downsample_ratio` as a Constant([0.5]) instead of a graph input.
     As a runtime input it forces Resize output to [?,?,?,?] which cascades:
     NPU then dies with INT64_MIN dims / 'rank 0 does not match filter rank 5'.
  4. Reshape inputs to fully static (480x640 + recurrent state shapes).
  5. convert_model + save fp16 IR into engine weights.

Usage:  python3 build_ir.py
Output: engine/src/models/weights/rvm-dsr05.xml / .bin
"""
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
from onnx import helper, numpy_helper, version_converter

HERE = Path(__file__).parent
ROOT = HERE.parent.parent  # studio-effects
SRC_ONNX = HERE / "rvm_mobilenetv3_fp32.onnx"
URL = "https://sourceforge.net/projects/robust-video-matting.mirror/files/v1.0.0/rvm_mobilenetv3_fp32.onnx/download"
OUT_IR = ROOT / "engine" / "src" / "models" / "weights" / "rvm-dsr05.xml"

DSR = 0.5
STATIC_INPUTS = {
    "src": [1, 3, 480, 640],
    "r1i": [1, 16, 120, 160],
    "r2i": [1, 20, 60, 80],
    "r3i": [1, 40, 30, 40],
    "r4i": [1, 64, 15, 20],
}


def ensure_source() -> Path:
    if SRC_ONNX.exists():
        return SRC_ONNX
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, SRC_ONNX)
    return SRC_ONNX


def build() -> None:
    ensure_source()

    print("loading onnx ...")
    model = onnx.load(str(SRC_ONNX))

    print("upgrading opset -> 17 ...")
    model = version_converter.convert_version(model, 17)

    # --- bake downsample_ratio as Constant, drop it from graph inputs ---
    graph = model.graph
    keep = [i for i in graph.input if i.name != "downsample_ratio"]
    removed = len(graph.input) - len(keep)
    del graph.input[:]
    graph.input.extend(keep)
    print(f"dropped downsample_ratio from inputs ({removed})")

    if not any(n.output == ["downsample_ratio"] for n in graph.node):
        graph.node.append(
            helper.make_node(
                "Constant",
                [],
                ["downsample_ratio"],
                value=numpy_helper.from_array(np.array([DSR], np.float32), name="dsr"),
            )
        )
        print(f"inserted Constant downsample_ratio={DSR}")

    baked = HERE / "rvm_dsr05.onnx"
    onnx.save(model, str(baked))

    print("converting to OpenVINO IR (static) ...")
    ov_model = ov.convert_model(str(baked))
    ov_model.reshape(STATIC_INPUTS)
    OUT_IR.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(OUT_IR), compress_to_fp16=True)
    print(f"saved {OUT_IR} (+ .bin)")

    # smoke test on NPU
    core = ov.Core()
    compiled = core.compile_model(str(OUT_IR), "NPU")
    req = compiled.create_infer_request()
    feeds = {
        "src": np.random.rand(1, 3, 480, 640).astype(np.float32),
        **{k: np.zeros(v, np.float32) for k, v in STATIC_INPUTS.items() if k != "src"},
    }
    out = req.infer(feeds)
    pha = out["pha"]
    print(f"NPU smoke OK — pha shape={pha.shape} mean={float(np.mean(pha)):.4f}")


if __name__ == "__main__":
    build()
