# Real-ESRGAN x4 — Image Super-Resolution

**Source**: https://github.com/xinntao/Real-ESRGAN
**License**: Apache-2.0
**Model type**: ONNX → OpenVINO IR

## Architecture

| Input  | `[1, 3, H, W]`    | Low-res image (3-channel BGR)          |
|--------|-------------------|---------------------------------------|
| Output | `[1, 3, H*4, W*4]`| 4× upscaled image                     |

## Conversion

```bash
# Export Real-ESRGAN to ONNX (PyTorch)
# Then:
mo --input_model realsr.onnx --data_type FP16 \
   --input_shape "[1,3,128,128]" \
   --output_dir ~/.local/share/studio-effects/models/realesrgan-x4
```

## Video pipeline

1. Capture frame from webcam (640×480)
2. Resize to 128×128 → Real-ESRGAN → 512×512 upscaled
3. Blend with original for real-time preview
4. Output to v4l2loopback `/dev/video10`

## Quality slider (0-100)

| Range  | Meaning                              |
|--------|--------------------------------------|
| 0-25   | Fast path (fewer diffusion steps)    |
| 26-50  | Balanced (10 steps)                  |
| 51-75  | High quality (15 steps)              |
| 76-100 | Max quality (20 steps)               |

## Performance

- **NPU**: ~40 ms/frame at 128×128 → 512×512
- **GPU fallback**: ~12 ms/frame
- **CPU fallback**: ~80 ms/frame
- **Burst workload**: only runs when "snapshot" is taken (not continuous)
