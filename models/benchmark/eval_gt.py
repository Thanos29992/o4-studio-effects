"""Ground-truth matting benchmark: MODNet (old) vs RVM (new).

Dataset: PPM-100 (portrait matting, human-annotated alpha mattes).
Both models run exactly as production (NPU, same preprocessing + postprocessing).
Metrics vs GT at 640x480:
  MAE%       mean |pred - gt| over all pixels        — overall matte error
  Edge MAE%  same, only GT in [0.05, 0.95]           — hair / transition quality
  FG-miss%   GT person but pred says background      — face would get blurred
  BG-leak%   GT background but pred says person      — bg detail stays sharp
"""
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parent.parent.parent  # studio-effects
DATASET = Path(__file__).parent / "PPM-100"
MODNET_PATH = ROOT / "engine" / "src" / "models" / "weights" / "modnet-fp16.onnx"
RVM_PATH = ROOT / "engine" / "src" / "models" / "weights" / "rvm-dsr05.xml"

W, H = 640, 480
REC_SHAPES = [(1, 16, 120, 160), (1, 20, 60, 80), (1, 40, 30, 40), (1, 64, 15, 20)]


def load_pairs():
    pairs = []
    for split in ("val", "train"):
        fg_dir = DATASET / split / "fg"
        for fg_path in sorted(fg_dir.glob("*.jpg")):
            gt_path = DATASET / split / "alpha" / fg_path.name
            if gt_path.exists():
                pairs.append((fg_path, gt_path))
    return pairs


def modnet_alpha(req, frame_bgr):
    # production-old path: 256x256 -> blur(3) -> (a-0.3)/0.4 clip
    resized = cv2.resize(frame_bgr, (256, 256))
    x = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = req.infer({0: x})
    alpha = list(out.values())[0].squeeze()
    alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return np.clip((alpha - 0.3) / 0.4, 0.0, 1.0).astype(np.float32)


def rvm_alpha(req, recs, frame_bgr):
    # production-new path: 640x480 -> blur(5) -> (a-0.05)/0.9 clip, states cycled
    resized = cv2.resize(frame_bgr, (W, H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    src = rgb.transpose(2, 0, 1)[None]
    for _ in range(5):  # warm recurrent states (steady state)
        out = req.infer({"src": src, "r1i": recs[0], "r2i": recs[1], "r3i": recs[2], "r4i": recs[3]})
        recs = [out["r1o"], out["r2o"], out["r3o"], out["r4o"]]
    alpha = out["pha"].squeeze()
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    return np.clip((alpha - 0.05) / 0.9, 0.0, 1.0).astype(np.float32)


def metrics(pred, gt):
    err = np.abs(pred - gt)
    fg = gt >= 0.5
    bg = ~fg
    edge = (gt > 0.05) & (gt < 0.95)
    pred_fg = pred >= 0.5
    return {
        "mae": err.mean() * 100,
        "edge_mae": err[edge].mean() * 100 if edge.any() else 0.0,
        "fg_miss": ((~pred_fg) & fg).sum() / max(fg.sum(), 1) * 100,
        "bg_leak": (pred_fg & bg).sum() / max(bg.sum(), 1) * 100,
    }


def main():
    core = ov.Core()
    core.set_property({"CACHE_DIR": str(Path(__file__).parent / "ov_cache")})

    print("loading models...", flush=True)
    mod_model = core.read_model(MODNET_PATH)
    mod_model.reshape((1, 3, 256, 256))  # production always runs static
    modnet = core.compile_model(mod_model, "NPU")
    mod_req = modnet.create_infer_request()
    print("modnet ready", flush=True)

    rvm_model = core.read_model(RVM_PATH)
    rvm = core.compile_model(rvm_model, "NPU")
    rvm_req = rvm.create_infer_request()
    print("rvm ready", flush=True)

    pairs = load_pairs()
    print(f"pairs: {len(pairs)}")

    agg = {"modnet": [], "rvm": []}
    t_mod, t_rvm = [], []

    for i, (fg_path, gt_path) in enumerate(pairs):
        frame = cv2.imread(str(fg_path))
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or gt is None:
            continue
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        t0 = time.perf_counter()
        a_mod = modnet_alpha(mod_req, frame)
        t_mod.append((time.perf_counter() - t0) * 1000)

        recs = [np.zeros(s, np.float32) for s in REC_SHAPES]
        t0 = time.perf_counter()
        a_rvm = rvm_alpha(rvm_req, recs, frame)
        t_rvm.append((time.perf_counter() - t0) * 1000)

        agg["modnet"].append(metrics(a_mod, gt))
        agg["rvm"].append(metrics(a_rvm, gt))
        print(f"[{i+1}/{len(pairs)}] {fg_path.name}: "
              f"modnet MAE={agg['modnet'][-1]['mae']:.2f}  rvm MAE={agg['rvm'][-1]['mae']:.2f}")

    print("\n" + "=" * 78)
    print(f"{'':18s} {'MAE%':>8s} {'Edge MAE%':>10s} {'FG-miss%':>9s} {'BG-leak%':>9s} {'ms/frame':>9s}")
    print("-" * 78)
    for name, lat in (("modnet", t_mod), ("rvm", t_rvm)):
        rows = agg[name]
        n = len(rows)
        print(f"{name:18s} "
              f"{np.mean([r['mae'] for r in rows]):8.2f} "
              f"{np.mean([r['edge_mae'] for r in rows]):10.2f} "
              f"{np.mean([r['fg_miss'] for r in rows]):9.2f} "
              f"{np.mean([r['bg_leak'] for r in rows]):9.2f} "
              f"{np.mean(lat):9.1f}")
    print("=" * 78)
    print("(all metrics lower = better; lat includes postprocess)")


if __name__ == "__main__":
    main()
