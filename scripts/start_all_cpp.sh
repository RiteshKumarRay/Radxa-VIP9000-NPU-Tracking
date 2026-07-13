#!/bin/bash
# start_all_cpp.sh — Starts NPU pipeline using the Native C++ application

echo "=============================================="
echo "  Starting Camera + C++ Native NPU Pipeline"
echo "=============================================="

pkill -9 npu_detect_cpp 2>/dev/null
pkill -9 gst-launch-1.0 2>/dev/null
pkill mediamtx 2>/dev/null
sleep 2

echo "[1/2] Starting MediaMTX WebRTC Server..."
cd /home/radxa
./mediamtx > /tmp/mediamtx.log 2>&1 &
sleep 2

IP=$(hostname -I | awk '{print $1}')
echo "      WebRTC  OK — http://$IP:8889/camera"

echo "[2/2] Starting C++ NPU Native Pipeline..."
export LD_LIBRARY_PATH=/home/radxa/npu
cd /home/radxa/npu
nohup ./npu_detect_cpp > /tmp/npu_detect_cpp.log 2>&1 &
NPU_PID=$!
sleep 3

if ! kill -0 $NPU_PID 2>/dev/null; then
    echo "ERROR: C++ NPU application failed to start!"
    echo "Log: /tmp/npu_detect_cpp.log"
    echo "Did you build it first? (cd npu_code && bash build_cpp.sh)"
    exit 1
fi

echo "      NPU OK — http://$IP:5000"
echo ""
echo "=============================================="
echo "  All services running! (C++ Engine Active)"
echo "=============================================="
echo ""
echo "  Camera WebRTC  : http://$IP:8889/camera"
echo "  NPU Web UI     : http://$IP:5000"
echo ""
echo "  Logs: /tmp/npu_detect_cpp.log"
echo "=============================================="
