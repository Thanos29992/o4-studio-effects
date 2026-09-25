# DeepFilterNet3 — Noise Suppression

**Source**: https://github.com/Rikorose/DeepFilterNet
**License**: MIT
**Model type**: ONNX (convert to OpenVINO IR)

## Model architecture (3 submodels)

| Submodel   | Input shape            | Output shape            | Purpose                          |
|------------|------------------------|-------------------------|----------------------------------|
| `enc`      | `[1, 3002, 32]`        | `[1, 3002, hidden]`     | ERB-domain encoder               |
| `erb_dec`  | `[1, 3002, 96]`        | `[1, 3002, 96]`         | ERB-domain decoder               |
| `df_dec`   | `[1, 3002, 96]`        | `[1, 3002, 96]`         | DF output / spectral floor mask  |

## Shapes for NPU (static before `compile_model`)

The NPU requires static shapes fixed before compilation. DeepFilterNet3 uses:
- **Frame size**: 2048 (FFT bins: 1025)
- **Hop size**: 256 (at 48kHz = ~5.3ms/chunk)
- **ERB bands**: 96 (DeepFilterNet's ERB filter bank count)
- **Batch**: 1 (single-mic, real-time inference)

## Conversion to OpenVINO IR

```bash
# 1. Extract ONNX from PyTorch
pip install deepfilter --no-deps
python -c "from deepfilter import DeepFilterNet; m = DeepFilterNet(); m.save_onnx('deepfilternet3.onnx')"

# 2. Convert to FP16 IR with static shapes
mo --input_model deepfilternet3.onnx \
   --data_type FP16 \
   --input_shape "[1,3002,32],[1,3002,96],[1,3002,96]" \
   --output_dir ~/.local/share/studio-effects/models/deepfilternet3

# 3. (Optional) Quantize to INT8 for lower NPU bandwidth
pot -c quantize_config.json -d deepfilternet3.xml
```

## Audio pipeline

1. PipeWire captures mic audio at 48kHz mono
2. STFT: 2048-point FFT, 256-sample hop, Hann window
3. ERB filterbank: 1025 bins → 96 ERB bands
4. OpenVINO inference: `enc` → `erb_dec` → `df_dec` (stateless per-frame)
5. ISTFT: reconstruct denoised waveform
6. PipeWire posts frames to a null sink → virtual mic for apps

## Performance

- **NPU**: ~3 ms/frame (stateless, per-frame inference)
- **CPU fallback**: ~15-20 ms/frame
- **Latency**: < 10 ms end-to-end at 48kHz
- **Memory**: ~40 MB model weights (FP16)

## NPU constraints

- Static shapes required before `compile_model("NPU")`
- FP16 IR only (no FP32 on NPU)
- Stateless per-frame models only (no RNN hidden state carried across frames
  — DeepFilterNet3's conv layers are per-frame compatible)

## Benchmark

```bash
studio-effects --record-test  # captures 5s from mic → processed output
# Compare before/after waveforms numerically:
python -c "
import numpy as np, soundfile as sf
a, sr = sf.read('test_raw.wav')
b, _ = sf.read('test_processed.wav')
print(f'noise reduction: {10*np.log10(np.var(a-b)/np.var(a)):.1f} dB')
"
```
