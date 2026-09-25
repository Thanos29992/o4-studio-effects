# Linux Studio Effects — Planning Document

> "I genuinely wanted to have that, you know, Windows Studio Effect kind of features in Linux as well."
> "I want to install it, yes, but I want to basically, you know, like do it my way."
> "Could we also add another model which basically you know does something extra like for example for boys [audio] we could add you know something that enhances the voice"
> "What if we could use AI models to make you know my cheap hardware seem like they are expensive ones"
> "Let's not execute this entire thing tonight because I want to sleep."

**Target hardware**: Intel Core Ultra 5 226V (Lunar Lake), 40 TOPS NPU ("Intel AI Boost"), Arc 130V iGPU.
**Target runtime**: C/C++ (no Python), OpenVINO for NPU targeting, with CPU/GPU fallbacks.
**Target UX**: Replicates the `shlok.asr` plugin/daemon architecture exactly — Quickshell bar panel + lightweight daemon with SIGUSR1 toggle + systemd user service + state files in `~/.local/share/studio-effects/state/`.

---

## 0. Architecture Summary (mirrors `shlok.asr` verbatim)

| Piece | Path | Pattern copied from |
|---|---|---|
| Quickshell bar plugin | `~/.config/omarchy/plugins/shlok.studio/manifest.json` + `Panel.qml` + `Model.js` + `assets/*.svg` | `shlok.asr` — same `Panel { moduleName: "shlok.studio" }`, same `Process` + `StdioCollector` state-file polling, same `BarIconButton` + `KeyboardPanel` layout, same "POWERED BY [intel.svg]" credit row |
| Floating popup | `~/.config/omarchy/plugins/shlok.studio-popup/RecorderPopup.qml` | `shlok.asr-popup/RecorderPopup.qml` — same `PanelWindow` on `WlrLayer.Overlay`, same 100ms/1200ms poll, same cancel-X via `cancel-requested` file |
| Daemon | `~/.local/bin/studio-effects --daemon` (C++ binary) | `dictate.cpp` — single process, `SIGUSR1` toggle, 100ms `select()` poll loop, `write_state()` to `status.json`, fork-per-session |
| Wrapper script | `~/.local/bin/omarchy-studio-effects {toggle\|start\|stop\|status}` | `omarchy-npu-dictate` — PID file, `kill -SIGUSR1`, daemon management |
| Systemd unit | `~/.config/systemd/user/omarchy-studio-effects.service` | `omarchy-npu-dictate.service` — `ExecStart=%h/.local/bin/studio-effects --daemon`, `Environment=WAYLAND_DISPLAY=…`, `Environment=LD_LIBRARY_PATH=…` |
| Hyprland bind | `~/.config/hypr/bindings.lua` — `o.bind("SUPER + SHIFT + code:201", "Studio", "omarchy-studio-effects toggle")` | same Copilot key reuse |
| State contract | `~/.local/share/studio-effects/state/` | identical file layout: `enabled`, `device.txt`, `model.txt`, `status.json`, `offload`, `volume`, `devices.json`, `level`, `daemon.pid`, `cancel-requested` |

---

## 1. Feature Matrix — Base Features

### 1.A Audio Noise Suppression (DeepFilterNet3)

**Model**: `Intel/deepfilternet-openvino` → DeepFilterNet3 IR (`.xml`/`.bin`) from the `deepfilternet3.zip` package on HuggingFace. The Arno500 fork (`Arno500/DeepFilterNet`) has already done the ONNX-to-OpenVINO graph surgery for NPU.

- **Size**: `enc.xml` ~3.5 MB, `erb_dec.xml` ~1.2 MB, `df_dec.xml` ~3.2 MB → total ~8 MB IR.
- **Input shape**: static `[1, 3002, 32]` for `feat_erb` and `[1, 2, 3002, 96]` for `feat_spec` (the Audacity plugin hardcodes `_num_hops = 3002` — this is 48 kHz × 10.24s / 160 hop, reshaped static for NPU).
- **Pipeline**: 48 kHz audio → STFT (50 ms, hop 10 ms) → ERB features `[1,1,3002,32]` → encoder (`feat_spec [1,2,3002,96]`) → erb_dec + df_dec → deep filter mask → ISTFT → 48 kHz enhanced audio → downsample to 16 kHz for passthrough.

| Stage | Model | Device | Rationale |
|---|---|---|---|
| `enc` (spectral + erb features → lsnr, encoder embeddings) | `enc.xml` | **NPU** | Small convnet; NPU-compiled via the Arno500 fork. ~3 ms/frame on NPU. |
| `erb_dec` (ERB-domain mask) | `erb_dec.xml` | **NPU** | Conv + reshape; NPU-compatible (Arno500-verified). |
| `df_dec` (sub-band deep filter coefficients) | `df_dec.xml` | **NPU** | Conv + reshape; NPU-compatible. |
| STFT / ISTFT / post-filter / resampling | — | **CPU** | Not a neural op; native C++ via kissFFT or libswresample. Runs inline in the ffmpeg/LADSPA pipeline. |

**Key insight from the Arno500 fork**: DeepFilterNet3's three ONNX models each get statically reshaped to `[1, 3002, ...]` before NPU compilation, with `ov::cache_dir()` set so the compiled blobs cache to disk. Runtime is ~11× less CPU than CPU-only on NPU, and the model is **stateless per-frame** (no LSTM carry-over across inference calls — each frame's features are self-contained), which is NPU-friendly.

**Audio wiring**: Use PipeWire `filter-chain` LADSPA plugin (the Arno500 fork ships one), or build a custom `studio-effects` C++ daemon that opens the mic via ALSA/PipeWire, runs the model via OpenVINO on the selected device, and writes processed audio to a v4l2loopback or PipeWire virtual sink/source. For the daemon pattern, follow `dictate.cpp`'s fork-per-session discipline: spawn a child per "effect session" (i.e., per recording window), load DeepFilterNet3 IR inside the child, process, unload.

**State file**: `model.txt` → `deepfilternet3`, `device.txt` → `NPU` (or `CPU`/`GPU` fallback). The daemon reads these at session start and compiles to the selected device.

**Controls**: ON/OFF toggle only (master). Config locked post-calibration — no sliders.

---

### 1.B Background Blur (MODNet Webcam v2 / RMBG v1.4)

**Model**: `Xenova/modnet` (ONNX, `modnet_photographic.onnx`) or `briaai/RMBG-1.4` (PyTorch → OpenVINO IR). For NPU compatibility, use the **ONNX-converted** version reshaped to static `[1,3,256,256]` (MODNet was in the OpenVINO Model Zoo docs explicitly; RMBG was verified in the OpenVINO notebooks).

- **Size**: MODNet ~25 MB ONNX, RMBG-1.4 IR ~170 MB.
- **Input**: RGB `[1,3,256,256]`, FP16 for NPU.
- **Output**: alpha matte `[1,1,256,256]` (values 0–1), same spatial dims as input.

| Stage | Model | Device | Rationale |
|---|---|---|---|
| Foreground segmentation (alpha matte) | `modnet_photographic.onnx` OR `rmbg-1.4.xml` | **NPU** | MobileNetV2 backbone, all conv ops; NPU-compiled at static 256×256. ~23 ms/frame on NPU (matching the `shlok.asr` NPU search findings). |
| Alpha-to-blur composite | — | **CPU/GPU** | Per-pixel Gaussian blur on the background region only. A single separable 15-tap blur on a 256×256 region is ~0.5 ms on CPU; composite via blend shader. |

**Pipeline**: webcam frame (e.g. 640×480) → resize to 256×256 → `modnet` on NPU → alpha matte → upscale matte to 640×480 (bilinear) → blur original frame's background region → composite foreground (sharp) + blurred background → output to v4l2loopback.

**NPU shape constraint**: MODNet/RMBG must be **statically reshaped to [1,3,256,256]** before `compile_model("NPU")`. The NPU rejects dynamic batch or spatial dims (exact same constraint verified in `feats.h`'s encoder path in `dictate.cpp:331-346`).

**v4l2loopback output**: The daemon writes processed frames (YUV/RGB) into `/dev/videoX` (a v4l2loopback device created via `modprobe v4l2loopback`). Zoom/Meet/Teams see a "virtual webcam" — same pattern as `linux-studio-effects` (ericjchang's project), but driven by our own daemon instead of an OBS plugin.

**Controls**: ON/OFF toggle + Blur strength slider (0–100 = kernel radius) + Background image picker (`state/bg_image_path.txt`).

---

### 1.C Auto-Framing (face-detection-adas-0001 + EMA smoothing)

**Model**: `face-detection-adas-0001` from OpenVINO Model Zoo / `Intel`'s published IR. MobileNetV2-SSD, trained for front-facing cameras.

- **Size**: ~5 MB (FP16 IR).
- **Input**: `[1,3,256,256]` BGR, FP16.
- **Output**: detection boxes `[1,1,N,7]` where each detection = `[image_id, label, conf, x_min, y_min, x_max, y_max]`.

| Stage | Model | Device | Rationale |
|---|---|---|---|
| Face detection | `face-detection-adas-0001.xml` | **CPU** | ~8–9 ms/frame on the 226V's P-cores at 256×256. The ericjchang/linux-studio-effects project runs this at ~9 ms/frame on CPU. NPU is possible but face detection is sparse (one call per frame, not sustained compute) — CPU is the pragmatic choice (matches user's "NPU 0%, CPU 5%" observation). |
| Bounding-box smoothing | — | **CPU** | Exponential Moving Average (alpha=0.04) on center + size — pure arithmetic. |
| Crop + pan/zoom | — | **CPU/GPU** | Resize + pan/zoom via OpenCV or a GLSL shader on the GPU. |

**Pipeline**: webcam frame → `face-detection-adas-0001` on CPU (256×256 input) → if face detected, EMA-smooth the bounding box → compute crop rect (centered on face, zoom_margin=2.2× like linux-studio-effects) → scale crop to output resolution → write to v4l2loopback.

**Controls**: ON/OFF toggle only. No sliders.

**Tuning** (from linux-studio-effects config the user will recognize):
```toml
[effects.auto_frame]
enabled = true
zoom_margin = 2.2
smoothing_factor = 0.08
detection_interval = 1   # run face detection every frame
```

**Key constraint**: Auto-framing is **not NPU-bound** — it's a low-compute detection + geometric transform. Running it on CPU keeps the NPU free for the sustained-compute effects (background blur). This matches the user's observation that their NPU was at 0% while streaming models ran on GPU.

---

## 2. Feature Matrix — Enhancement Features ("DLSS-style")

> "what if we could use AI models to make you know my cheap hardware seem like they are expensive ones"

### 2.A Audio Super-Resolution / Voice Enhancement (AudioSR)

**Model**: `Intel/versatile_audio_super_resolution_openvino` → the `.zip` packages from HuggingFace contain `audio_sr_encoder`, `audio_sr_decoder`, `vae_feature_extract`, `vocoder`, and DDPM stage as OpenVINO IR.

- **Basic model** (`basic`) or **Speech model** (`speech`) — two checkpoint variants.
- **Pipeline stages**: encoder → VAE feature extract → DDPM (diffusion) → decoder → vocoder.
- **Output**: 48 kHz, 24-bit audio (upscaled from the mic's native ~16 kHz).

**How the NPU + GPU split works (simple)**: AudioSR is a 5-stage pipeline. The NPU is powerful enough to run the **DDPM** (the heaviest AI denoising stage) but cannot run the **encoder**, **decoder**, or **vocoder** — Intel's own Audacity plugin code (`OVAudioSR.cpp`) explicitly skips NPU for those three stages because they use tensor ops the NPU compiler can't handle. So the daemon uses two `ov::Core` instances: one targeting NPU for the DDPM, one targeting GPU for the other four stages, and pipes data between them on the host.

| Stage | IR files | Device | Rationale |
|---|---|---|---|
| Encoder + VAE feature extract | `audio_sr_encoder`, `vae_feature_extract` | **GPU** | STFT + convolutional feature extraction. NPU-incompatible per Intel's code. |
| DDPM (diffusion denoiser) | `audio_sr_ddpm_speech` / `audio_sr_ddpm_basic` | **NPU** | Conv + attention denoiser — NPU-compatible (confirmed via realesrgan/yolov3 in the OpenVINO NPU model list). |
| Decoder + Vocoder | `audio_sr_decoder`, `vocoder` | **GPU** | Transposed conv + HiFi-GAN-style vocoder — NPU-incompatible per Intel's code. |

**Key constraint**: This is a **chunked pipeline**, not a real-time stream. AudioSR processes 10.24-second or 5.12-second chunks (matching the Audacity plugin's `chunk_size` config). The DDPM stage takes ~1.3s per 10.24s chunk on the NPU (Intel's benchmark, 25 DDIM steps). So this enhancement feature is **not a live microphone effect** — it's a **record-then-enhance** processor:

1. User enables AudioSR in the panel → daemon starts recording from the selected mic/source into `capture_raw.wav`.
2. On stop (2nd tap), the daemon slices the recording into 10.24s chunks, runs each through the full pipeline (encoder→GPU, DDPM→NPU, decoder→GPU, vocoder→GPU).
3. Enhanced 48 kHz audio is written to `state/enhanced.wav` and can be replayed via the localhost test page or a virtual audio sink.

**State file**: `model.txt` → `audiosr-speech`, `device.txt` → `NPU` (the daemon internally maps DDPM→NPU, encoder/decoder/vocoder→GPU automatically).

**Controls**: ON/OFF toggle + Quality preset dropdown (Low / Med / High = DDIM steps 10/25/50) + Real-time checkbox (live mic passthrough vs record-then-enhance).

---

### 2.B Image Upscaling / Webcam Enhancement (Real-ESRGAN ×4)

**Model**: `Intel/real-esrgan-x4` or the OpenVINO-optimized `realesrgan-x4` from the OpenVINO Model Hub. Already confirmed NPU-supported (per the OpenVINO ONNX EP docs: `realesrgan-x4` is in the NPU model list).

- **Size**: ~3.5 MB FP16 IR (the x4 generator).
- **Input**: `[1,3,64,64]` tiled patches (ESRGAN uses overlapping tiles for large images).
- **Output**: `[1,3,256,256]` upscaled patches → stitched.

| Stage | Model | Device | Rationale |
|---|---|---|---|
| Real-ESRGAN x4 generator | `realesrgan-x4.xml` | **NPU** | All conv + pixel-shuffle ops; NPU-compatible; ~23 ms/tile at 256×256 matching the Studio Effects budget. |
| Tile stitching / color clamp | — | **CPU/GPU** | Pure arithmetic. |

**Pipeline**: webcam frame (e.g. 640×480) → split into 64×64 overlapping tiles (16px overlap) → run each tile through Real-ESRGAN ×4 on NPU → each tile outputs 256×256 → stitch tiles back into a 2560×1920 enhanced frame → downscale back to 640×480 (bilinear) → this is the "sharpness enhancement" step → output to v4l2loopback.

This is a **cheap-hardware-enhancement** trick: the webcam stays at native resolution, but the Real-ESRGAN pass removes compression artifacts and sharpens the image before downscaling, making a $20 webcam look like a $200 one. It's the exact same idea as NVIDIA Broadcast's "super-resolution" enhancement.

**State file**: `model.txt` → `realesrgan-x4`, `device.txt` → `NPU`.

**Controls**: ON/OFF toggle + Single quality slider (0–100 controls scaling factor + denoise strength + tile overlap).

---

## 3. State File Contract (exact copy of `shlok.asr`)

```
~/.local/share/studio-effects/state/
├── enabled          → "on"/"off" (master toggle)
├── device.txt       → "CPU"/"GPU"/"NPU" (accelerator selection)
├── model.txt        → "deepfilternet3" / "modnet" / "face-adas" / "audiosr-speech" / "realesrgan-x4"
├── status.json      → {"alt":"idle","class":"recording|idle","tooltip":"","stream":1?,"since":epoch}
├── offload          → "1 30" (immediate) or "0 <secs>" (keep in RAM)
├── volume           → integer 0-100 (audio feedback / effect intensity slider)
├── level            → plain decimal RMS (for VU bar in popup)
├── devices.json     → {"models":{"deepfilternet3":["NPU","CPU","GPU"],"modnet":["NPU","CPU","GPU"],...}}
├── cancel-requested → empty file (popup X button)
├── daemon.pid       → PID of running daemon
└── capture.wav      → raw captured frame/audio (fallback)
```

---

## 4. Plugin Panel Design (`Panel.qml` — mirrors `shlok.asr` exactly)

### Bar Icon Glyphs
- **Idle**: `󰋧` (U+F267, mdi-imac) — neutral "effects" icon, dimmed when disabled.
- **Processing (frame in flight)**: `󰏚` (U+F3DA, mdi-cog-outline) — spinning gear.
- **Error**: `󰀨` (U+F066, mdi-alert) — if a model fails to load on the selected device.

### Panel Sections (simplified per user spec — keep controls lean):
1. **Hero row**: icon · "STUDIO EFFECTS" · Master ToggleSwitch (on/off, controls ALL effects).
2. **Accelerator row**: CPU | GPU | NPU (SVG chips, same `assets/cpu.svg`/`gpu.svg`/`npu.svg`, same `assetForDevice()` pattern). Dimmed/filtered per model compatibility via `devices.json`. "POWERED BY" + `intel.svg` credit row.
3. **Effect toggles** (per-feature ON/OFF + controls below each):
   - **Noise Suppression**: ON/OFF toggle. When off, the daemon bypasses DeepFilterNet3 entirely (dead path). No sliders — config locked after first good calibration.
   - **Background Blur**: ON/OFF toggle + Blur strength slider (0–100) + Background image picker (file dialog → `state/bg_image_path.txt`). When off, original webcam feeds through untouched.
   - **Auto-Framing**: ON/OFF toggle only. When off, passthrough. No sliders.
   - **Voice SR (AudioSR)**: ON/OFF toggle + Quality preset dropdown (Low/Med/High = DDIM steps 10/25/50). Real-time passthrough mode (checkbox) vs record-then-enhance (radio toggle).
   - **Image SR (Real-ESRGAN)**: ON/OFF toggle + Single quality slider (0–100 controls scaling factor / denoise strength / tile overlap).
4. **Model offload slider**: same 8-stop ramp (Immediate / 30s / 1m / 2m / 5m / 10m / 15m / Never). Maps to `state/offload`.
5. **Audio input device selector**: dropdown of PipeWire source nodes (from `devices.json`/`state/audio_source.txt`). Auto-handled by PipeWire graph re-routing — switching input device in GNOME Settings keeps the effect active automatically.

### `Model.js` — additions to the ASR pattern:
```js
var EFFECT_DISPLAY_NAMES = {
  "deepfilternet3": "Noise Suppression",
  "modnet": "Background Blur",
  "face-adas": "Auto-Framing",
  "audiosr-speech": "Voice SR (AudioSR)",
  "realesrgan-x4": "Image Upscale (Real-ESRGAN)"
}

var EFFECT_VENDOR_GLYPHS = {
  "deepfilternet3": "",  // Intel (noise suppression)
  "modnet": "󰂰",           // background removal
  "face-adas": "󰀀",          // face
  "audiosr-speech": "",    // audio
  "realesrgan-x4": "󰈁"       // camera
}

function effectLabel(name) {
  return EFFECT_DISPLAY_NAMES[name] || Model.modelLabel(name)
}
```

---

## 5. Build System (`build.sh`)

```bash
#!/bin/bash
# ~/.local/share/studio-effects/src/build.sh
set -e
BIN=$HOME/.local/bin/studio-effects
SRC=$HOME/.local/share/studio-effects/src

g++ -O2 -std=c++17 -Wall -Wextra \
  -I/usr/include/openvino \
  -I$SRC \
  $SRC/studio_effects.cpp \
  $SRC/feats.h \
  -lopenvino -lsndfile -lfftw3f -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
  -lv4l2 -lpthread \
  -o "$BIN"
chmod +x "$BIN"
echo "BUILD_OK"
```

**Key libraries**:
- `libopenvino.so` — model inference on NPU/GPU/CPU.
- `libfftw3f` — STFT/FFT for audio features (reused from `feats.h`, byte-compatible).
- `libsndfile` — WAV read/write (for AudioSR input/output).
- `libopencv_*` — image resize, tile stitching, Gaussian blur, crop (no Qt dependency).
- `libv4l2` — write to v4l2loopback device.
- `libpipeWire` — capture from PipeWire source.

**Compile-time constraint**: The NPU plugin requires FP16 IR models. Download them via `omz_downloader` and convert to IR:
```bash
omz_downloader --name modnet-photographic-portrait-matting --output_dir models/
omz_converter --name modnet-photographic-portrait-matting --output_dir models/ --data_type FP16
```

---

## 6. Audio Input Device Selection

> "was it so because allowing the model to work with any microphone would cause problems somehow, or the OEM or the manufacturer only wired the shit to the default laptop's mic and didnt care"

**Answer**: OEM/driver limitation, not a technical one. Windows Studio Effects uses `IntelAudioFilterDriver.sys` which registers as a Windows audio effect in the system driver stack — but OEMs only certified the built-in mic array in their WHQL submission. External USB/Thunderbolt mics use a different USB audio driver stack that the Intel filter driver doesn't hook into. On Linux with PipeWire, we bypass this entirely.

**Design decision**: The daemon will read the source device from `state/audio_source.txt` (written by the panel). Default is `@DEFAULT_AUDIO_SOURCE@` (the system's default mic). The user can change it to any PipeWire node name:

- `@DEFAULT_AUDIO_SOURCE@` — system default (laptop mic, USB mic, whatever is selected in GNOME Settings)
- `alsa_input.usb-Logitech_...` — any named PipeWire source
- `alsa_input.pci-0000_00_1f.3...` — built-in laptop mic (explicit)

**Panel addition**: A dropdown section "MICROPHONE" shows `pw-cli ls Node 1004 1 2 3` output (source nodes with class "audio/Source"), lets the user pick one. Writes to `state/audio_source.txt`. The daemon reads it at session start and passes `--record` to PipeWire with the right node.

**For v4l2loopback + camera effects**: Same pattern — `state/video_source.txt` holds the PipeWire/video node (usually `@DEFAULT_VIDEO_SOURCE@`). The daemon opens that source and writes processed frames to `/dev/video10`.

## 7. Testing Approach — Localhost Test Page + Record/Replay

### 7.A Local Web Test Page for Camera Effects

The user explicitly wants a localhost web page to test camera effects in real time. This is the only way to verify visual effects (you can't debug background blur from a log line).

**Design**: `studio-effects test-camera` launches a local HTTP server (embedded C++ or `python3 -m http.server`) on port 18080 that serves:
1. A `navigator.mediaDevices.getUserMedia({video: {deviceId: "<source>"}})` stream from the v4l2loopback device (`/dev/video10`) — this shows what the *output* of our effects pipeline looks like.
2. Optionally, a split-view showing the raw webcam source next to the processed `/dev/video10` feed.

**File**: `~/.local/share/studio-effects/test-page/camera.html` — a single HTML file with inline JS and CSS (no npm/node dependency). Uses vanilla JS `RTCPeerConnection` or `MediaSource` to pipe the webcam + v4l2loopback into two `<video>` elements side-by-side.

**Port**: Always `http://localhost:18080`. The panel shows a persistent "Open Test Page" button that does `xdg-open http://localhost:18080`.

**Implementation choice**: The daemon serves the page via a built-in HTTP server (use a minimal C HTTP library or a tiny embedded Python script forked from the daemon). The page is always available when the daemon is running — no manual server start needed.

### 7.B Audio Record + Replay

Audio effects (noise suppression, AudioSR) can't be visually inspected — they must be heard. The testing approach:

1. **Record**: `studio-effects --record-test` captures 10 seconds of raw mic audio → `test_raw.wav`.
2. **Apply**: Runs the selected effect (DeepFilterNet3 noise suppression OR AudioSR voice SR) on the raw file.
3. **Output**: Writes `test_processed.wav`.
4. **Replay**: `studio-effects --play-test` plays both `test_raw.wav` and `test_processed.wav` back-to-back via PulseAudio (or lets the user play them manually with `pw-play`).

**Delay measurement**: The daemon logs the total processing time per chunk:
- DeepFilterNet3: ~3 ms/frame × (10s × 1000 frames/s) = ~30 ms total for 10 seconds of audio → negligible (<1 frame at 48 kHz).
- AudioSR: ~1.3s per 10.24s chunk → effectively real-time for the first chunk, but subsequent chunks pipeline.

**Panel UI**: A "TEST AUDIO" section with two buttons: "Record 10s" (writes `test_raw.wav`), then "Play Raw / Play Processed" (toggles between the two files via `pw-play`).

## 8. Implementation Phases

### Phase 1 — Foundation (daemon + plugin shell + test page)
- [ ] Create `~/.local/share/studio-effects/` directory structure (mirror `npu-asr/`).
- [ ] Copy the `shlok.asr` plugin files verbatim into `shlok.studio/`, rename `moduleName` to `shlok.studio`, change `productName` to "STUDIO EFFECTS", update glyph set.
- [ ] Copy `RecorderPopup.qml` into `shlok.studio-popup/`, update `namespace` to `omarchy-studio-effects`.
- [ ] Write `studio_effects.cpp` — the daemon skeleton (copy `dictate.cpp`'s daemon loop, `write_state()`, `read_state_str()`, signal handlers, PID file, 100ms `select()` poll, `cancel_requested()` / `clear_cancel()`). No models yet.
- [ ] Write the localhost test page: `test-page/camera.html` with split-view webcam + v4l2loopback display.
- [ ] Write `build.sh`, compile the skeleton daemon.
- [ ] Install systemd unit, wrapper script, Hyprland keybind.
- [ ] Wire `Panel.qml` to toggle the daemon's `enabled` state (SIGUSR1 → toggle).

### Phase 2 — Audio Noise Suppression (DeepFilterNet3)
- [ ] Download `Intel/deepfilternet-openvino` IR for DeepFilterNet3.
- [ ] Implement `DeepFilterNet3::load(device)` — static reshape `[1,3002,32]` / `[1,2,3002,96]`, `compile_model(device)`, cache via `ov::cache_dir`.
- [ ] Implement `DeepFilterNet3::enhance(float* pcm_48k, size_t n)` — STFT → features → 3 OpenVINO inferences → deep filter mask → ISTFT.
- [ ] Wire PipeWire capture (mic) → 48 kHz → DeepFilterNet3 → 48 kHz enhanced → resample to 16 kHz → write to a PipeWire virtual source / v4l2loopback for audio.
- [ ] Test: `studio-effects --test-audio` plays processed mic through headphones.
- [ ] Panel: `model.txt` → `deepfilternet3`, `device.txt` → `NPU` (default).

### Phase 3 — Background Blur (MODNet) + v4l2loopback + test page
- [ ] Download `modnet-photographic-portrait-matting` ONNX → convert to FP16 IR, static `[1,3,256,256]`.
- [ ] Implement `Modnet::load(device)`, `Modnet::segment(RGB* frame_256_256)` → alpha matte.
- [ ] Implement composite: alpha-blend blur(original_bg_region) + sharp(fg_region).
- [ ] Create v4l2loopback device: `modprobe v4l2loopback video_nr=10 exclusive_vidcap=1`.
- [ ] Pipe webcam (PipeWire) → 256×256 → MODNet on NPU → upscale matte → blur+composite → write RGB to `/dev/video10`.
- [ ] Test: localhost test page shows split-view of raw webcam + blurred `/dev/video10` output.
- [ ] Panel: `model.txt` → `modnet`, `device.txt` → `NPU` (default).

### Phase 4 — Auto-Framing (Face Detection + EMA)
- [ ] Download `face-detection-adas-0001` FP16 IR, static `[1,3,256,256]`.
- [ ] Implement `FaceTracker::detect()` → bounding box → EMA smoothing (alpha=0.04).
- [ ] Implement crop + pan/zoom: center on face, zoom_margin=2.2×.
- [ ] Write cropped frame to `/dev/video10` (same loopback device — effects are mutually exclusive by design, like Windows Studio Effects).
- [ ] Test: localhost test page shows face tracking in action.
- [ ] Panel: `model.txt` → `face-adas`, `device.txt` → `CPU` (face detection stays on CPU — NPU is reserved for sustained compute).

### Phase 5 — Enhancement Features (Audio + Image SR)

#### 5a Voice Super-Resolution (AudioSR)
- [ ] Clone `Intel/versatile_audio_super_resolution_openvino` IR files.
- [ ] Implement `AudioSR::load()` with **per-stage device routing**: encoder+VAE→GPU, DDPM→NPU, decoder+vocoder→GPU.
- [ ] Implement chunked processing: 10.24s slices, `ddim_steps=25` (balance quality/latency), `guidance_scale=3.5`.
- [ ] Wire: daemon captures raw mic into `state/capture_raw.wav`, on stop triggers AudioSR over the whole buffer, writes `state/enhanced.wav`.
- [ ] Test: record 10s → run AudioSR (~1.3s for the DDPM stage) → play raw vs enhanced side-by-side.
- [ ] Panel: `model.txt` → `audiosr-speech`, `device.txt` → `NPU` (DDPM stage only; daemon routes internally).

#### 5b Image Upscale (Real-ESRGAN ×4)
- [ ] Download `realesrgan-x4` FP16 IR from OpenVINO Model Hub.
- [ ] Implement `RealESRGAN::load()`, `RealESRGAN::upscale(RGB* tile_64_64)` → 256×256 tile.
- [ ] Implement tile pipeline: split 640×480 into 64×64 tiles (16px overlap), upscale each on NPU, stitch → 2560×1920 → downscale to 640×480.
- [ ] Wire into the webcam pipeline: webcam → Real-ESRGAN tiles → downscale → output to `/dev/video10`.
- [ ] Test: localhost test page shows the Real-ESRGAN sharpness enhancement.
- [ ] Panel: `model.txt` → `realesrgan-x4`, `device.txt` → `NPU`.

### Phase 6 — Polish & Integration
- [ ] Live status poll in `Panel.qml` (800ms `ProcStatus` → `barGlyph` ternary, exact same pattern as ASR).
- [ ] VU bar + timer in `RecorderPopup.qml` (reused verbatim from ASR popup — RMS from `state/level`, timer from `status.json since`).
- [ ] Audio feedback: `start.wav`/`stop.wav` via the double-fork + `setsid` + `paplay` pattern from `dictate.cpp:590-641`, with the same piecewise-linear `volume_to_gain()` curve.
- [ ] `omarchy-studio-effects toggle` in `bindings.lua` (reuse the Copilot hardware key).
- [ ] Audio source selector dropdown (pipewire node scan → `state/audio_source.txt`).
- [ ] Video source selector dropdown (→ `state/video_source.txt`).
- [ ] `devices.json` writer: daemon scans model dirs at startup, reads `devices.txt`, writes `state/devices.json` for the panel to filter on.

---

## 8. Genuinely Feasible vs. Hard Limits

| Feature | NPU | GPU | CPU | Verdict |
|---|---|---|---|---|
| DeepFilterNet3 noise suppression | ✅ `enc`+`erb_dec`+`df_dec` (static shapes, stateless per-frame) | ✅ fallback | ✅ fallback | **Genuinely feasible on NPU.** Small conv models, no LSTM carry-over, ~3 ms/frame. |
| MODNet background blur | ✅ conv-only MobileNetV2 backbone | ✅ | ✅ | **Genuinely feasible on NPU.** ~23 ms/frame at 256×256, matches the budget. |
| Face detection (auto-framing) | ⚠️ possible but no gain | — | ✅ ~8–9 ms/frame | **CPU is the right call.** Detection is sparse (one call/frame); NPU is reserved for sustained compute. |
| AudioSR (voice SR) | ✅ DDPM stage only | ✅ encoder/decoder/vocoder | ✅ DDPM on CPU (slow) | **Partially real-time.** DDPM on NPU is the "sustained NPU workload" the user asked about. But it's chunked (10.24s slices), not a live mic effect. |
| Real-ESRGAN ×4 (image upscaling) | ✅ conv + pixel-shuffle, NPU model list confirmed | ✅ | ✅ | **Genuinely feasible on NPU.** Tile-based pipeline, ~23 ms/tile. |

### Hard limits / what will NOT work:

1. **Streaming AudioSR on NPU** — the DDPM + VAE encoder + vocoder pipeline requires multiple sequential model loads with 48 kHz → 24 kHz → 10.24s chunking. It's a **post-processing effect**, not a live passthrough. This is the fundamental difference from DeepFilterNet (which is a single inference per audio frame).

2. **NPU audio super-resolution with LSTM state** — the user's stated goal of "sustained NPU workload" is best served by DeepFilterNet3 and MODNet, both of which are stateless per-frame. AudioSR's DDPM is the closest thing to a "long-running NPU job" but it processes fixed 10.24s chunks, not a continuous stream.

3. **Dynamic shapes on NPU** — every model MUST be statically reshaped before `compile_model("NPU")`, exactly as `dictate.cpp` does for the Parakeet encoder (`np["audio_signal"] = ov::PartialShape{1, 128, 3001}`). This is the single most important pattern to copy.

4. **NPU + GPU simultaneously** — the 226V's NPU and Arc 130V iGPU are separate devices. OpenVINO can target both, but you can't run two different `CompiledModel`s on two devices in the same infer request. Multi-stage pipelines (like AudioSR) must use **separate `ov::Core` instances** per device and pipe data between them on the host — same pattern as the Audacity `OVAudioSR.cpp` which uses 3 device-selection dropdowns.

---

## 9. File Layout (complete, mirrored from `npu-asr`)

```
~/.local/share/studio-effects/
├── src/
│   ├── studio_effects.cpp      ← main daemon (copy dictate.cpp's daemon + signal handling)
│   ├── feats.h                 ← COPY from npu-asr (LogMel128, byte-compatible)
│   ├── audio_pipeline.hpp      ← DeepFilterNet3 STFT/ISTFT (C++/kissFFT)
│   ├── video_pipeline.hpp      ← MODNet composite, face tracking, Real-ESRGAN tiling
│   ├── audiosr.hpp             ← AudioSR chunked pipeline (encoder→DDPM→decoder→vocoder)
│   └── build.sh
├── models/
│   ├── deepfilternet3/         ← Intel/deepfilternet-openvino IR (8 MB)
│   ├── modnet/                 ← modnet IR (25 MB, FP16)
│   ├── face-detection-adas-0001/ ← OpenVINO Model Zoo IR (5 MB, FP16)
│   ├── audiosr-speech/         ← Intel/versatile_audio_super_resolution_openvino IR
│   ├── realesrgan-x4/          ← realesrgan-x4 IR (3.5 MB, FP16)
│   └── vad/                    ← silero_vad_v4.onnx (if needed)
├── cache/                      ← OpenVINO compiled-blob cache (set via ov::cache_dir)
├── state/                      ← runtime state files (exactly as shlok.asr)
├── sounds/
│   ├── start.wav               ← COPY from npu-asr
│   └── stop.wav
└── studio-effects.service      ← systemd unit (copy from npu-asr)

~/.config/omarchy/plugins/
├── shlok.studio/
│   ├── manifest.json           ← copy shlok.asr, change id→"shlok.studio", name→"Studio Effects"
│   ├── Panel.qml               ← copy shlok.asr/Panel.qml, update stateDir/modelsDir/glyphs
│   ├── Model.js                ← copy shlok.asr/Model.js, add EFFECT_DISPLAY_NAMES + effectLabel()
│   └── assets/
│       ├── cpu.svg / gpu.svg / npu.svg / intel.svg   ← COPY from shlok.asr
│       ├── intel.svg, nvidia.svg, openai.svg, huggingface.svg  ← COPY
│       └── studio.svg            ← NEW: effects/glyph icon
└── shlok.studio-popup/
    ├── manifest.json           ← declare barWidget → "popup" style
    └── RecorderPopup.qml         ← copy shlok.asr-popup, update namespace + namespace string

~/.config/hypr/bindings.lua
├── (unbind SUPER+SHIFT+code:201 from Dictate)
└── o.bind("SUPER + SHIFT + code:201", "Studio", "omarchy-studio-effects toggle")

~/.local/bin/
├── studio-effects              ← compiled C++ daemon
├── omarchy-studio-effects    ← wrapper script (copy from omarchy-npu-dictate, rename)
└── v4l2loopback-ctl           ← helper to modprobe/create /dev/video10

~/.config/systemd/user/
└── omarchy-studio-effects.service  ← copy from omarchy-npu-dictate.service, update paths
```

---

## 10. The NPU Workload Answer (directly addressing the user's question)

> "what genuinely would be good running continuously on NPU in background?"

From the `npu-workload-search-session` memory: the deciding constraint is workload **shape**, not speed. OCR was ruled out because it's one-shot (Python startup dominates). Streaming ASR (parakeet-v3) can't target NPU (GGUF → Vulkan only). Whisper-base works on NPU only because it's stateless one-shot.

**The answer for Studio Effects**: The two genuinely-sustained NPU workloads are:

1. **DeepFilterNet3 audio noise suppression** — `enc` + `erb_dec` + `df_dec` run continuously on incoming audio frames (3 ms/frame on NPU, stateless per-frame). This is the closest thing to a "sustained always-on NPU workload" you can build. It runs at ~3002 hops/chunk × 3 ms = sustained NPU utilization while the mic is active.

2. **MODNet background segmentation** — runs at ~23 ms/frame on NPU while the camera is active (the user themselves noted "background blur ~23 ms/frame at 256×256 is feasible on NPU" in their research). This is the video analog of DeepFilterNet: stateless per-frame, sustained while the camera stream flows.

These are the two features that will genuinely keep the NPU busy at 20–25% (matching the user's Windows experience: "voice 5%, camera effects 15%, total 20-25%"). Auto-framing (face detection) stays on CPU because it's sparse detection, not sustained compute. AudioSR's DDPM is the third option but it's chunked, not continuous.

**This is the NPU workload search result**: background blur (MODNet) + noise suppression (DeepFilterNet3) are your two always-on NPU workloads. Everything else (face detection, audio super-resolution, image upscaling) is either CPU-bound, chunked, or bursty.

---

## 11. Sub-Plan: GitHub Publishing Workflow (`o4-studio-effects`)

> "i ALSO wanna make my professional profiles being built and steadily growing as im exploring my hobbies and shit lol"

This sub-plan runs **in parallel** to the main implementation plan. The goal is incremental, professional-looking GitHub commits that build your profile as you implement each feature. Not everything gets pushed — only relevant, clean files.

### Repository: `o4-studio-effects`
- **Name**: `o4-studio-effects` (o4 = Omarchy + 4th-gen, or whatever feels right)
- **Public repo** on GitHub under your profile
- **README**: Personal, captures the ethos — "custom-defined Studio Effects for Linux, targeting the Intel Core Ultra NPU, inspired by Windows Studio Effects"
- **Push discipline**: One commit per completed phase, README updated only after a significant milestone is verified working.

### `.gitignore` rules (what stays local, what goes public):

```
# NEVER push — local-only state
state/*                     # runtime state files (enabled, device.txt, model.txt, etc.)
state/!*.sample             # except template samples
*.pid
*.wav                       # test recordings
test_raw.wav
test_processed.wav
capture*.wav
enhanced.wav
daemon.pid
cancel-requested

# Cache blobs (13GB+ — never push)
cache/
*.blob

# Compiled binaries (build artifacts — let users build from source)
studio-effects
*.o
*.so

# Omarchy system integration (local to your machine, not portable)
~/.config/hypr/bindings.lua   # user keybinding (your personal config)

# Environment-specific
*.service                     # systemd units have %h paths, not portable
```

### What DOES get pushed:

| File/Dir | Why | When |
|---|---|---|
| `src/*.cpp`, `src/*.hpp`, `src/*.h` | The daemon C++ source — the core of the project | Phase 1 (skeleton) + each phase as features land |
| `src/build.sh` | Build instructions so others can compile | Phase 1 |
| `omarchy/plugins/shlok.studio/` | Quickshell plugin QML + JS + manifest | Phase 1 |
| `omarchy/plugins/shlok.studio-popup/` | Popup QML | Phase 1 |
| `omarchy/plugins/shlok.studio-assets/` | SVG icons, shared across plugin + popup | Phase 1 |
| `omarchy/bindings.lua.sample` | Template showing the keybind pattern (not your personal config) | Phase 6 |
| `tools/` (test-page/camera.html, helpers) | Localhost test page + utility scripts | Phase 1 + as needed |
| `models/*.txt` (NOT the .xml/.bin files) | Model download URLs + checksums, not binaries | Phase 2-5 as each model lands |
| `docs/SETUP.md` | Installation + v4l2loopback setup guide | Phase 6 |
| `docs/HARDWARE.md` | Your target hardware specs (builds your profile narrative) | Phase 1 |
| `LICENSE` | MIT or GPL — open source your work | Phase 1 |
| `README.md` | Updated only after milestones | After Phases 1, 2, 4, 6 |

### Commit + README schedule (milestone-driven):

1. **After Phase 1** (foundation): Commit skeleton daemon + plugin shell + test page. README lists the project, hardware, build instructions, "what this is."
2. **After Phase 2** (DeepFilterNet3): Commit noise suppression. README updates with "Noise Suppression working on Intel NPU."
3. **After Phase 3** (MODNet blur): Commit background blur. README adds "Background Blur working on Intel NPU."
4. **After Phase 4** (face tracking): Commit auto-framing. README adds "Auto-Framing (CPU)."
5. **After Phase 5** (AudioSR + Real-ESRGAN): Commit enhancements. README adds AudioSR + Image SR.
6. **After Phase 6** (polish): Commit final wiring. README updated to reflect complete feature set, full hardware list, performance notes.

### Hardware section for README/docs/HARDWARE.md:

```markdown
## Target Hardware

| Component | Specification |
|---|---|
| Laptop | Acer Aspire 14 AI 52-MT |
| CPU | Intel Core Ultra 5 226V (Lunar Lake) |
| GPU | Intel Arc 130V (integrated) |
| NPU | Intel AI Boost NPU (4-bit, ~40 TOPS) |
| RAM | 16GB LPDDR5X 8533MT/s (soldered) |
| OS | Arch Linux (Omarchy) + Hyprland + Quickshell |
```

### Directory structure for the repo:

```
o4-studio-effects/
├── README.md
├── LICENSE
├── docs/
│   ├── SETUP.md
│   └── HARDWARE.md
├── src/
│   ├── studio_effects.cpp
│   ├── studio_effects.hpp
│   ├── audio_pipeline.hpp
│   ├── video_pipeline.hpp
│   ├── audiosr.hpp
│   ├── realesrgan.hpp
│   ├── facesr.hpp
│   ├── face_tracker.hpp
│   ├── feats.h          # copied from npu-asr
│   ├── build.sh
│   └── download_models.sh
├── omarchy/
│   ├── plugins/shlok.studio/
│   │   ├── manifest.json
│   │   ├── Panel.qml
│   │   ├── Model.js
│   │   └── assets/
│   │       ├── cpu.svg
│   │       ├── gpu.svg
│   │       ├── npu.svg
│   │       └── intel.svg
│   ├── plugins/shlok.studio-popup/
│   │   ├── manifest.json
│   │   └── RecorderPopup.qml
│   └── bindings.lua.sample
├── tools/
│   ├── test-page/
│   │   └── camera.html
│   ├── setup_v4l2loopback.sh
│   ├── download_all_models.sh
│   └── monitor_npu.sh
└── models/
    ├── deepfilternet3.md      # download URL + checksum + shape
    ├── modnet.md
    ├── face-detection-adas-0001.md
    ├── audiosr-speech.md
    └── realesrgan-x4.md
```

### Commit message style:
```
feat(audio): add DeepFilterNet3 noise suppression on NPU

- Static reshape [1,3002,32] / [1,2,3002,96] before compile_model("NPU")
- STFT→enc→erb_dec→df_dec→ISTFT pipeline via OpenVINO
- PipeWire source → processed virtual sink
- Panel toggle + state/audio_source.txt support

Benchmark: ~3ms/frame on NPU, zero latency at 48kHz
```
