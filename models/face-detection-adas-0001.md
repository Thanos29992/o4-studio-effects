# Face Detection ADAS 0001 — Auto-Framing

**Source**: Intel Open Model Zoo
**License**: Apache-2.0

## Architecture

| Input  | `[1, 3, 384, 384]` | BGR image           |
|--------|--------------------|--------------------|
| Output | `[1, 1, 208, 7]`   | Detection results  |

## Download

```bash
omz_download_tool --name face-detection-adas-0001 \
  --output_dir ~/.local/share/studio-effects/models/face-adas
omz_converter --name face-detection-adas-0001 \
  --output_dir ~/.local/share/studio-effects/models/face-adas
```

## Auto-framing logic

1. Face detector runs every ~8 frames (sparse, not per-frame)
2. Bounding box → center crop with 1.5× margin
3. Smooth transitions with exponential moving average
4. CPU-only: face detection is sparse, not a sustained NPU workload

## Performance

- **CPU**: ~15 ms/detection at 384×384
- **Interval**: every 320ms (8 frames at 25 FPS)
- **NPU**: possible but overkill — stays on CPU by design
