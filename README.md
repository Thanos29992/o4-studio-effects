# o4-studio-effects

Studio Effects for Linux.

I built this because Windows has Studio Effects — noise suppression, background blur, auto-framing — that work out of the box with that laptop's NPU. On Linux, nothing like that exists at the system level. So I'm writing one from scratch.

## Hardware

This runs on my Acer Aspire 14 AI laptop:

- **CPU**: Intel Core Ultra 5 226V
- **GPU**: Intel Arc 130V (integrated)
- **NPU**: Intel AI Boost (~40 TOPS)
- **RAM**: 16GB LPDDR5X

The NPU is the interesting part — it's always on, uses almost no power, and can handle small AI models at decent speed. That's what I'm targeting for the effects.

## What it does

Five effects, split into two groups:

**Always-on effects** (these run continuously while the mic/cam are active):
1. **Noise suppression** — DeepFilterNet3 on NPU. Cleans up background noise from audio.
2. **Background blur** — MODNet on NPU. Blurs your background in video calls.
3. **Auto-framing** — Face detection on CPU. Keeps your face centered in the frame.

**On-demand enhancements** (these get applied to a recording, not live):
4. **Voice super-resolution** — AudioSR. Upscales voice audio from 16kHz to 48kHz.
5. **Image upscaling** — Real-ESRGAN ×4. Sharpens webcam video.

## How it works

- A C++ daemon runs in the background (same pattern as my ASR project).
- A Quickshell panel in the Omarchy bar lets you toggle effects on/off.
- Models load on-demand, use the NPU, and unload when done.
- Audio goes through PipeWire; video goes through v4l2loopback.

This is not a plugin for OBS or a Python script. It's a native system-level effect that works across all apps — Zoom, Discord, Google Meet — without them knowing anything about it.

## Status

Just started. Code is coming phase by phase. See [PLAN.md](PLAN.md) for the full roadmap.
