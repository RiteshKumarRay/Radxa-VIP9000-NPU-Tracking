#!/bin/bash
# start_all.sh — Starts NPU pipeline + WebRTC Server

echo "=============================================="
echo "  Starting Camera + NPU Person Detection"
echo "=============================================="

pkill -9 python3 2>/dev/null
pkill -9 gst-launch-1.0 2>/dev/null
pkill mediamtx 2>/dev/null
sleep 2

echo "[1/2] Starting MediaMTX WebRTC Server..."
cd /home/radxa
./mediamtx > /tmp/mediamtx.log 2>&1 &
sleep 2

IP=$(hostname -I | awk '{print $1}')
echo "      WebRTC  OK — http://$IP:8889/camera"

echo "[2/2] Starting NPU person detection + Camera Pipeline..."
export LD_LIBRARY_PATH=/home/radxa/npu
nohup /home/radxa/yolo_env/bin/python3 /home/radxa/npu/npu_detect.py > /tmp/npu_detect.log 2>&1 &
NPU_PID=$!
sleep 5

if ! kill -0 $NPU_PID 2>/dev/null; then
    echo "ERROR: NPU detection failed to start!"
    echo "Log: /tmp/npu_detect.log"
    exit 1
fi

echo "      NPU OK — http://$IP:5000"
echo ""
echo "=============================================="
echo "  All services running!"
echo ""
echo "  Camera WebRTC  : http://$IP:8889/camera"
echo "  NPU Detection  : http://$IP:5000"
echo "  NPU MJPEG feed : http://$IP:5000/video_feed"
echo ""
echo "  Logs:"
echo "    Camera/NPU : /tmp/npu_detect.log"
echo "=============================================="
