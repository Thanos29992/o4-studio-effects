#!/bin/bash
# Live monitor: CPU / GPU / NPU / RAM usage in one view.
# Run: ./tools/monitor_usage.sh  (Ctrl+C to stop)
#
# Helps verify which accelerator a model actually uses.
# If NPU says 0% but you expected it to fire, the model fell back to CPU/GPU.

NPU_PATH="/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us"

# Detect GPU idle path (may vary by system)
GPU_BASE=""
for base in /sys/class/drm/card*/device/tile0; do
    if [ -f "$base/gt0/gtidle/idle_residency_ms" ]; then
        GPU_BASE="$base"
        break
    fi
done

# Detect v4l2loopback devices
V4L_DEVICES=$(ls /dev/video* 2>/dev/null | head -10 | tr '\n' ' ')

while true; do
    # --- CPU ---
    read -r _ u1 n1 s1 i1 w1 q1 sq1 st1 _ < /proc/stat
    # --- GPU ---
    gpu_busy="N/A"
    if [ -n "$GPU_BASE" ]; then
        a0=$(cat "$GPU_BASE/gt0/gtidle/idle_residency_ms" 2>/dev/null)
        a1=$(cat "$GPU_BASE/gt1/gtidle/idle_residency_ms" 2>/dev/null)
    fi
    # --- NPU ---
    npu_a=$(cat "$NPU_PATH" 2>/dev/null)
    npu_t=$(date +%s%N)
    # --- RAM ---
    mem=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "%.1f", (t-a)/1024/1024}' /proc/meminfo)

    sleep 1

    # --- CPU (end) ---
    read -r _ u2 n2 s2 i2 w2 q2 sq2 st2 _ < /proc/stat
    t1=$((u1+n1+s1+i1+w1+q1+sq1+st1))
    t2=$((u2+n2+s2+i2+w2+q2+sq2+st2))
    idle1=$((i1+w1))
    idle2=$((i2+w2))
    cpu_busy=$(awk -v t1="$t1" -v t2="$t2" -v i1="$idle1" -v i2="$idle2" '
    BEGIN { d=t2-t1; di=i2-i1; printf "%.1f", (d-di)/d*100 }')

    # --- GPU (end) ---
    if [ -n "$GPU_BASE" ]; then
        b0=$(cat "$GPU_BASE/gt0/gtidle/idle_residency_ms" 2>/dev/null)
        b1=$(cat "$GPU_BASE/gt1/gtidle/idle_residency_ms" 2>/dev/null)
        delta0=$((b0 - a0))
        delta1=$((b1 - a1))
        busy0=$((100 - delta0 * 100 / 1000))
        busy1=$((100 - delta1 * 100 / 1000))
        gpu_busy=$((busy0 > busy1 ? busy0 : busy0))
        gpu_busy="${gpu_busy}%"
    fi

    # --- NPU (end) ---
    npu_b=$(cat "$NPU_PATH" 2>/dev/null)
    npu_t2=$(date +%s%N)
    if [ -n "$npu_a" ] && [ -n "$npu_b" ] && [ "$npu_a" != "$npu_b" ]; then
        busy=$((npu_b - npu_a))
        elapsed=$(( (npu_t2-npu_t)/1000 ))
        npu_busy=$(awk -v busy="$busy" -v elapsed="$elapsed" '
        BEGIN { printf "%.1f", (busy / elapsed) * 100 }')
    else
        npu_busy="N/A"
    fi

    # --- Print ---
    printf "\r\033[KCPU: %s%%  GPU: %s  NPU: %s%%  RAM: %sGB  v4l2: %s" \
        "$cpu_busy" "$gpu_busy" "$npu_busy" "$mem" "$V4L_DEVICES"
done
