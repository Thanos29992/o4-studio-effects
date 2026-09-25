# AudioSR — Voice Super-Resolution

**Source**: https://github.com/haoheliu/audiosr
**License**: Apache-2.0
**Model type**: ONNX → OpenVINO IR (5-stage pipeline)

## Architecture

AudioSR is a 5-stage diffusion pipeline:
1. **Vocoder (1)**: 1D CNN → coarse features
2. **Coarse (2)**: 2D U-Net (256×24 → 256×80, 32kHz)
3. **Fine (3)**: 2D U-Net denoiser (256×80 → 256×240)
4. **Discriminator (4)**: GAN discriminator
5. **Postnet (5)**: 1D CNN → waveform (48kHz output)

## Key constraints

- **Chunked, not continuous**: AudioSR processes 4-second chunks (not per-frame)
- **Batch**: `[1, 1, T]` where T = frames (48000 * 4 = 192000)
- **Two Core instances**: AudioSR on GPU + DeepFilterNet3 on NPU simultaneously
  (requires separate `ov::Core` per pipeline)
- **5 DDPM timesteps**: ~100ms per chunk on NPU

## Conversion

```bash
# AudioSR is PyTorch → export to ONNX → convert to IR
python export_audiosr.py --output_dir ~/.local/share/studio-effects/models/audiosr-speech
mo --input_model audiosr.onnx --data_type FP16 --output_dir ~/.local/share/studio-effects/models/audiosr-speech
```

## Pipeline

1. Mic → 4-second chunk buffer
2. On chunk full: AudioSR enhances → output to speaker + recording
3. Interleaved with DeepFilterNet3 (live noise suppression on stream)

## Presets

| Preset | Chunk size | Timesteps | Quality | NPU time |
|--------|-----------|-----------|---------|----------|
| low    | 2s        | 20        | 0.7x    | ~30ms    |
| med    | 4s        | 32        | 1.0x    | ~100ms   |
| high   | 4s        | 50        | 1.3x    | ~200ms   |
