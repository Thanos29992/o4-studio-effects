#!/bin/bash
# Setup v4l2loopback for virtual camera output.
# Creates /dev/video10 as a virtual camera that apps can read from.
# The video pipeline writes processed frames to this device.
#
# Usage: ./tools/setup_v4l2loopback.sh (start|stop|status)
#
set -euo pipefail

VIDEO_DEV="/dev/video10"
CARD_LABEL="Studio Effects"

check_module() {
    lsmod | grep -q v4l2loopback && return 0 || return 1
}

start_virtualcam() {
    if [ -e "$VIDEO_DEV" ]; then
        echo "Virtual camera already exists at $VIDEO_DEV"
        return 0
    fi

    if ! check_module; then
        echo "[1/3] Loading v4l2loopback module..."
        modprobe v4l2loopback devices=1 video_nr=10 card_label="$CARD_LABEL" exclusive_caps=1 2>/dev/null || {
            echo "ERROR: failed to load v4l2loopback module"
            echo "Install: sudo pacman -S v4l2loopback-dkms"
            exit 1
        }
    fi

    if [ ! -e "$VIDEO_DEV" ]; then
        # Module loaded but device not created
        rmmod v4l2loopback 2>/dev/null || true
        modprobe v4l2loopback devices=1 video_nr=10 card_label="$CARD_LABEL" exclusive_caps=1
    fi

    echo "[2/3] Created virtual camera: $VIDEO_DEV ($CARD_LABEL)"

    # Verify
    if [ -e "$VIDEO_DEV" ]; then
        echo "[3/3] OK: $(v4l2-ctl --list-devices 2>/dev/null | grep -A1 "$CARD_LABEL" || true)"
        echo ""
        echo "Apps can now select '$CARD_LABEL ($VIDEO_DEV)' as their camera source."
    else
        echo "ERROR: $VIDEO_DEV was not created"
        exit 1
    fi
}

stop_virtualcam() {
    if check_module; then
        echo "Removing v4l2loopback module..."
        rmmod v4l2loopback 2>/dev/null || true
        echo "Virtual camera removed."
    else
        echo "v4l2loopback not loaded."
    fi
}

status_virtualcam() {
    if [ -e "$VIDEO_DEV" ]; then
        echo "Virtual camera: $VIDEO_DEV (active)"
        v4l2-ctl --list-formats-ext --device "$VIDEO_DEV" 2>/dev/null || true
    else
        echo "Virtual camera: not available ($VIDEO_DEV)"
    fi
}

case "${1:-start}" in
    start)  start_virtualcam ;;
    stop)   stop_virtualcam ;;
    status) status_virtualcam ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
