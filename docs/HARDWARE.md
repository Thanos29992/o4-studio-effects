## Target Hardware

| Component | Specification |
|---|---|
| Laptop | Acer Aspire 14 AI 52-MT |
| CPU | Intel Core Ultra 5 226V (Lunar Lake) |
| GPU | Intel Arc 130V (integrated) |
| NPU | Intel AI Boost NPU (~40 TOPS, 4-bit quantization) |
| RAM | 16GB LPDDR5X 8533MT/s (soldered) |
| OS | Arch Linux (Omarchy) + Hyprland + Quickshell |

### Development Environment Notes

- The NPU is stateless-only — every model must be statically reshaped before `compile_model("NPU")`.
- OpenVINO 2026.x is the inference backend for all NPU/GPU/CPU targets.
- Audio flows through PipeWire; video flows through PipeWire → v4l2loopback.
- The daemon uses the same fork-per-session + SIGUSR1 toggle pattern as `shlok.asr`.
