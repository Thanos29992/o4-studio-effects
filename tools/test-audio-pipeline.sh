#!/bin/bash
# Test the audio pipeline end-to-end (Phase 2 scaffolding).
# This script:
#   1. Generates a 5-second test signal (speech-like + noise)
#   2. Runs it through the pipeline (simulated — Phase 2 will use real PipeWire)
#   3. Compares before/after waveforms numerically
#   4. Cleans up after itself
#
set -euo pipefail

STATE_DIR="$HOME/.local/share/studio-effects/state"
BIN="$HOME/.local/bin/studio-effects"
TMP="/tmp/studio-effects-test.$$"

mkdir -p "$TMP"

echo "=== Studio Effects Audio Pipeline Test ==="
echo ""

# 1. Generate test signal: 5 seconds of 440Hz tone + white noise
echo "[1/4] Generating test signal (440Hz tone + noise)..."
python3 -c "
import numpy as np, soundfile as sf
sr = 48000
t = np.linspace(0, 5, 5*sr, endpoint=False)
tone = 0.3 * np.sin(2*np.pi*440*t)
noise = np.random.randn(len(t)) * 0.5
mix = tone + noise
sf.write('$TMP/test_raw.wav', mix, sr)
print(f'[gen] 5s signal at {sr}Hz, noise level: {np.std(noise)/np.std(tone):.2f}')
" 2>&1

# 2. Simulate pipeline (Phase 2: real inference will go here)
echo "[2/4] Running through pipeline (simulated)..."
cp "$TMP/test_raw.wav" "$TMP/test_processed.wav"

# 3. Compare before/after
echo "[3/4] Analyzing before/after..."
python3 -c "
import numpy as np, soundfile as sf

raw, sr = sf.read('$TMP/test_raw.wav')
proc, _ = sf.read('$TMP/test_processed.wav')

# Metrics
noise_power_before = np.var(raw[:1000])  # first 1000 samples
noise_power_after = np.var(proc[:1000])

print(f'  Signal RMS (raw):      {np.sqrt(np.mean(raw**2)):.4f}')
print(f'  Signal RMS (processed): {np.sqrt(np.mean(proc**2)):.4f}')
print(f'  Shape match:           {raw.shape == proc.shape}')
print(f'  Sample rate:           {sr} Hz')
" 2>&1

# 4. Cleanup
echo "[4/4] Cleaning up..."
rm -rf "$TMP"
echo ""
echo "=== Test complete ==="
echo "Phase 2: Replace simulated pipeline with:"
echo "  - PipeWire source capture (pactl load-module + pw-stream)"
echo "  - DeepFilterNet3 NPU inference (OpenVINO)"
echo "  - Virtual sink output (pw-loopback)"
