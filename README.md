# VIP9000 NPU YOLOv5 Person Detection Streamer (Radxa)

This project provides a hardware-accelerated person detection pipeline for Radxa boards equipped with the Vivante VIP9000 NPU. It captures video from a camera, runs a YOLOv5s model on the NPU using a custom C wrapper (`awnn_shim.c`), and streams the annotated MJPEG video via a Flask server.

## Features
- **Hardware-accelerated Inference**: Utilizes the VIP9000 NPU via `libNBGlinker.so` and `libVIPhal.so`.
- **YOLOv5s**: Uses a pre-compiled YOLOv5s `.nb` model optimized for the T527/VIP9000.
- **Low Latency**: ~50ms inference time on the NPU.
- **Web Stream**: Hosts a live MJPEG stream with bounding boxes and FPS overlay using Flask.

## Project Structure
```text
.
├── npu_code/
│   ├── awnn_shim.c      # C wrapper to interface with VIP9000 NPU
│   ├── build.sh         # Build script for the C wrapper (compile on board)
│   ├── npu_detect.py    # Main Flask app & YOLOv5 inference logic
│   ├── vip_lite.h       # VIP9000 NPU Header
│   └── yolov5s.nb       # Compiled YOLOv5s model for VIP9000
├── scripts/
│   ├── start_all.sh     # Script to start MediaMTX, GStreamer, and the Python app
│   └── start_camera.sh  # Script to start the RTSP stream using GStreamer
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Setup & Installation (On Radxa Board)

### 1. Compile the NPU Wrapper
The Python script needs `libawnn_npu.so` to communicate with the NPU. You must compile this on the Radxa board.
```bash
cd npu_code
bash build.sh
```
This will generate `libawnn_npu.so`.

### 2. Install Python Dependencies
It's recommended to use a virtual environment.
```bash
python3 -m venv yolo_env
source yolo_env/bin/activate
pip install -r requirements.txt
```

### 3. Setup the Camera Stream
Ensure you have `mediamtx` installed and running. The detection script pulls an RTSP stream (e.g., `rtsp://127.0.0.1:8554/camera`).
You can use the provided `scripts/start_camera.sh` to begin a hardware-encoded GStreamer pipeline to feed MediaMTX.

## Running the Application

You can use the wrapper script to start everything automatically:
```bash
bash scripts/start_all.sh
```

Or run the Python app manually:
```bash
cd npu_code
export LD_LIBRARY_PATH=/home/radxa/npu
python3 npu_detect.py
```

Then, open your browser and navigate to:
`http://<RADXA_IP>:5000`

## Important Notes
- The model `yolov5s.nb` is required for inference. It has been exported correctly with 3 FP32 outputs, bypassing the aggressive quantization issues found in some YOLOv8 exports for this architecture.
- The `npu_detect.py` script applies sigmoid functions and standard YOLOv5 anchor decoding to the raw NPU output tensors.
