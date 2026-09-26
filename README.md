# o4-studio-effects

> **Linux Studio Effects** — AI-powered noise suppression, background blur, auto-framing, voice super-resolution, and image upscaling for Linux.
>
> Targeted at Intel Core Ultra processors with NPU ("Intel AI Boost") + Arc iGPU.
> Runs on NPU, GPU, or CPU with OpenVINO.
> **Auto-activates** when your mic or camera is in use — no buttons needed.

---

## What is this?

A reimplementation of Windows Studio Effects for Linux, built on OpenVINO and PipeWire. Unlike manual-toggle solutions, effects load **automatically** when a camera or mic stream is active:

| Effect | Model | Accelerator | Notes |
|--------|-------|-------------|-------|
| **Noise Suppression** | DeepFilterNet3 | NPU (fallback: GPU/CPU) | Real-time, ~3ms/frame on NPU |
| **Background Blur** | RVM (MobileNetV3) | NPU (fallback: GPU) | ~14ms/frame at 640×480, GT-benchmarked (MAE 1.3% vs MODNet 3.1%) |
| **Auto-Framing** | face-detection-adas-0001 | CPU | Sparse detection, not continuous |
| **Voice Super-Resolution** | AudioSR | GPU + NPU (dual-core) | Chunked, 4s windows |
| **Image Upscaling** | Real-ESRGAN x4 | GPU (fallback: NPU) | On-demand for snapshots |

## Architecture

The system has two parts:

1. **Daemon** (`studio-effects --daemon`) — Event-driven, low-power (~0.1% CPU when idle). Sleeps in `select()` until woken by:
   - Inotify events (panel toggles an effect ON/OFF)
   - PipeWire stream events (camera/mic starts/stops — Phase 2+)

   When a stream is active and the corresponding effect is enabled in the panel, the daemon auto-loads the model and processes the stream. No manual start/stop button.

2. **Panel** (`shlok.studio` Quickshell plugin) — Bar icon opens a control panel with per-effect ON/OFF toggles and parameter sliders. The system ON/OFF is managed by `systemctl --user` (daemon start/stop), not a panel toggle.

## Target Hardware

| Component | Specification |
|---|---|
| Laptop | Acer Aspire 14 AI 52-MT |
| CPU | Intel Core Ultra 5 226V (Lunar Lake) |
| GPU | Intel Arc 130V (integrated) |
| NPU | Intel AI Boost NPU (40 TOPS, 4-bit) |
| OS | Arch Linux (Omarchy) + Hyprland + Quickshell |

## Setup

### Prerequisites

```bash
# Arch (via Omarchy)
omarchy pkg add openvino pipewire-alsa pipewire-jack v4l2loopback-dkms
```

### Install

```bash
# Clone
git clone https://github.com/Thanos29992/o4-studio-effects.git ~/.local/share/studio-effects

# Build daemon
cd ~/.local/share/studio-effects
./src/build.sh

# Copy panel to Omarchy plugins
mkdir -p ~/.config/omarchy/plugins
cp -r omarchy/plugins/shlok.studio ~/.config/omarchy/plugins/
cp -r omarchy/plugins/shlok.studio-popup ~/.config/omarchy/plugins/

# Initialize state files from templates
cp state/*.sample ~/.local/share/studio-effects/state/
# Remove .sample extension
cd ~/.local/share/studio-effects/state
for f in *.sample; do mv "$f" "${f%.sample}"; done

# Install systemd service
cp ~/.local/share/studio-effects/omarchy-studio-effects.service \
   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now omarchy-studio-effects

# Create virtual camera
modprobe v4l2loopback devices=1 video_nr=10 card_label="Studio Effects"
```

### Download Models

```bash
cd ~/.local/share/studio-effects
./src/download_models.sh all
```

## Usage

1. **Start the daemon**: `systemctl --user start omarchy-studio-effects`
2. **Open the panel**: Click the Studio Effects icon in the bar
3. **Enable effects**: Toggle any effect ON in the panel
4. **Use your mic/camera**: The daemon auto-detects active streams and applies the corresponding effect

## Testing

```bash
# Audio pipeline test (generates synthetic signal, analyzes RMS)
./tools/test-audio-pipeline.sh

# Camera test (opens split-view HTML page)
python3 -m http.server 18080 --directory tools/test-page &
# Then open: http://localhost:18080/camera.html

# Live resource monitor
./tools/monitor_usage.sh
```

## License

MIT — see [LICENSE](LICENSE).

Models downloaded separately; their licenses apply to model weights only.
