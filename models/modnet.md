# MODNet — Background Segmentation (Background Blur)

**Source**: https://github.com/saic-vul/BackgroundMattingV2 (MODNet variant)
**License**: Apache-2.0
**Model type**: ONNX → OpenVINO IR

## Model architecture

| Input  | `[1, 3, 256, 256]` | Image tensor (BGR, normalized)            |
|--------|---------------------|------------------------------------------|
| Output | `[1, 1, 256, 256]` | Segmentation mask (alpha matte)           |

## Conversion

```bash
mo --input_model modnet.onnx \
   --data_type FP16 \
   --input_shape "[1,3,256,256]" \
   --output_dir ~/.local/share/studio-effects/models/modnet
```

## Video pipeline

1. v4l2loopback creates `/dev/video10` (virtual camera)
2. PipeWire captures raw webcam feed
3. Each frame: 256×256 resize → MODNet inference → alpha matte
4. Composite: foreground (alpha-blended) + background (Gaussian blur)
5. Output to v4l2loopback for apps (Zoom, OBS, etc.)

## Performance

- **NPU**: ~23 ms/frame at 256×256 (2026.3 IR on 4TB NPU)
- **GPU fallback**: ~8 ms/frame
- **CPU fallback**: ~40 ms/frame
- **Output res**: 640×480 (for real-time at ~25 FPS)
