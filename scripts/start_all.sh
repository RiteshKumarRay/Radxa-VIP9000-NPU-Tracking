#!/bin/bash
# start_all.sh — Starts camera pipeline + NPU person detection
# Usage: bash /home/radxa/start_all.sh

echo "=============================================="
echo "  Starting Camera + NPU Person Detection"
echo "=============================================="

# Kill any old instances
pkill -9 python3 2>/dev/null
pkill -9 gst-launch-1.0 2>/dev/null
pkill mediamtx 2>/dev/null
sleep 2

# Start camera pipeline with hardware H264 encoding
echo "[1/2] Starting 1080p 30fps hardware stream..."
sudo bash /home/radxa/start_camera.sh > /tmp/cam.log 2>&1 &
sleep 4

# Check camera started
if ! pgrep -x gst-launch-1.0 > /dev/null; then
    echo "ERROR: Camera pipeline failed to start!"
    echo "Log: /tmp/cam.log"
    exit 1
fi
echo "      Camera OK — rtsp://$(hostname -I | awk '{print $1}'):8554/camera"
echo "      WebRTC  OK — http://$(hostname -I | awk '{print $1}'):8889/camera"

# Start NPU detection Flask server
echo "[2/2] Starting NPU person detection..."
export LD_LIBRARY_PATH=/home/radxa/npu
nohup /home/radxa/yolo_env/bin/python3 /home/radxa/npu/npu_detect.py > /tmp/npu_detect.log 2>&1 &
NPU_PID=$!
sleep 5

# Check NPU started
if ! kill -0 $NPU_PID 2>/dev/null; then
    echo "ERROR: NPU detection failed to start!"
    echo "Log: /tmp/npu_detect.log"
    exit 1
fi

IP=$(hostname -I | awk '{print $1}')
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
echo "    Camera : /tmp/cam.log"
echo "    NPU    : /tmp/npu_detect.log"
echo "=============================================="
