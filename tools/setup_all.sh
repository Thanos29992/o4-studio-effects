#!/bin/bash
# One-shot setup for Studio Effects on a fresh machine.
# Run: ./tools/setup_all.sh
#
# This script:
#   1. Creates state directory + copies sample files as defaults
#   2. Installs the Omarchy Quickshell plugins (shlok.studio + popup)
#   3. Installs the systemd user service
#   4. Starts the daemon
#   5. Attempts to set up v4l2loopback virtual camera
#
set -euo pipefail

SRC_DIR="$HOME/.local/share/studio-effects"
STATE_DIR="$SRC_DIR/state"
SYSTEMD_DIR="$HOME/.config/systemd/user"
PLUGINS_DIR="$HOME/.config/omarchy/plugins"

echo "=== Studio Effects Setup ==="
echo ""

# 1. State directory
echo "[1/5] Initializing state directory..."
mkdir -p "$STATE_DIR"
for f in "$SRC_DIR"/state/*.sample; do
    base=$(basename "$f" .sample)
    if [ ! -f "$STATE_DIR/$base" ]; then
        cp "$f" "$STATE_DIR/$base"
        echo "  + $base"
    fi
done

# 2. Install plugins
echo "[2/5] Installing Omarchy plugins..."
mkdir -p "$PLUGINS_DIR"
cp -r "$SRC_DIR/omarchy/plugins/shlok.studio" "$PLUGINS_DIR/"
cp -r "$SRC_DIR/omarchy/plugins/shlok.studio-popup" "$PLUGINS_DIR/"
echo "  Installed: shlok.studio + shlok.studio-popup"

# 3. Systemd service
echo "[3/5] Installing systemd service..."
mkdir -p "$SYSTEMD_DIR"
cp "$SRC_DIR/omarchy-studio-effects.service" "$SYSTEMD_DIR/"
systemctl --user daemon-reload 2>/dev/null || true
echo "  Service: omarchy-studio-effects.service"

# 4. Start daemon
echo "[4/5] Starting daemon..."
systemctl --user enable omarchy-studio-effects 2>/dev/null || true
systemctl --user start omarchy-studio-effects 2>/dev/null || true
sleep 2
if systemctl --user is-active omarchy-studio-effects 2>/dev/null | grep -q active; then
    echo "  Daemon: RUNNING ✓"
else
    echo "  Daemon: FAILED to start"
    echo "  Try manually: $SRC_DIR/../bin/studio-effects --daemon"
fi

# 5. Virtual camera
echo "[5/5] Setting up virtual camera..."
if modprobe -n v4l2loopback &>/dev/null; then
    modprobe v4l2loopback devices=1 video_nr=10 card_label="Studio Effects" exclusive_caps=1
    if [ -e /dev/video10 ]; then
        echo "  v4l2loopback: /dev/video10 created ✓"
    else
        echo "  v4l2loopback: loaded but /dev/video10 not found"
        echo "  Install: sudo pacman -S v4l2loopback-dkms"
    fi
else
    echo "  v4l2loopback: module not available"
    echo "  Install: sudo pacman -S v4l2loopback-dkms"
    echo "  Then: sudo modprobe v4l2loopback devices=1 video_nr=10 card_label='Studio Effects'"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Open the panel: click the bar icon or run:"
echo "  omarchy-studio-effects status"
echo ""
echo "Test pages:"
echo "  Audio:     ./tools/test-audio-pipeline.sh"
echo "  Video:     python3 -m http.server 18080 --directory tools/test-page"
echo "  Monitor:   ./tools/monitor_usage.sh"
echo ""
echo "Logs:"
echo "  journalctl --user -u omarchy-studio-effects -f"
