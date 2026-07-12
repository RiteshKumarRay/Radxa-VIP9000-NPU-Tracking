# Radxa VIP9000 Hardware-Accelerated YOLOv5 Pipeline

This repository provides an ultra-optimized, zero-copy architecture for running a live camera stream and YOLOv5s real-time person detection on Radxa boards equipped with the Allwinner ISP and Vivante VIP9000 NPU (e.g., Radxa ZERO 3W, Cubie A7S).

## The Goal
The objective of this project was to achieve **Real-Time NPU Object Detection** (specifically YOLOv5s person detection) while simultaneously streaming a pristine, low-latency 1080p 30FPS camera feed over the web, without melting the CPU.

## The Problem: The Allwinner ISP Deadlock
If you try to build an NPU pipeline on these boards using standard tutorials, you will hit a massive wall. Specifically, the Allwinner ISP (`en-awisp=1`) and GStreamer's memory mapping interface have severe driver-level race conditions. 

**What we faced:**
1. **Kernel Panics & Lockups:** When attempting to capture video via OpenCV (`cv2.VideoCapture`), or when trying to map the GStreamer buffer into C/Python space, the pipeline deadlocks. Mutexes lock up (`__lll_lock_wait`), GStreamer throws `GST_IS_BUFFER` assertions, and the entire board freezes requiring a hard power cycle. (This is a [known issue in the Radxa community](https://forum.radxa.com/t/a7s-gstreamer-current-status-for-real-time-npu/31163)).
2. **Color Space Glitches:** Forcing the pipeline through `videoconvert` triggers NV12/YV12 buffer sizing mismatches, resulting in horrible green and purple tints.
3. **CPU Bottlenecks:** Forcing the CPU to handle video encoding maxes out the processor, severely throttling the NPU inference frame rate.

## The Solution: Dual-Branch `fdsink` Architecture
We abandoned the broken shared-memory mapping approach entirely and architected a custom, decoupled pipeline that bypasses the ISP driver bugs.

Instead of fighting GStreamer memory leaks in Python, we use a single, rock-solid GStreamer CLI subprocess (`gst-launch-1.0`) to split the hardware feed into two independent branches:

1. **Zero-Copy WebRTC (H.264):** One branch of the `tee` pipes pristine 1080p frames directly into the silicon `omxh264videoenc` hardware encoder. This streams to the web at 0% CPU load with perfect colors.
2. **Decoupled `fdsink` Capture:** The second branch converts the frames to flat `BGR` bytes and pipes them to `stdout` via `fdsink`. Python simply reads this byte-stream safely from standard output. Because we aren't mapping shared memory buffers, the kernel deadlocks are **physically impossible** in this architecture.

## What's in this Repository?
We have provided everything needed to bypass the bugs and get the Vivante VIP9000 working cleanly:

* `npu_sdk/` - Contains all the proprietary Allwinner/Vivante C-headers and `.so` shared libraries (`libNBGlinker.so`, `libVIPhal.so`, etc.) extracted directly from the Radxa SDK. You don't need a massive Docker container; everything is here.
* `npu_code/awnn_shim.c` - Our custom C-wrapper that communicates directly with the VIP9000 NPU, bridging the low-level C libraries to our Python server.
* `npu_code/yolov5s.nb` - The pre-compiled, optimized YOLOv5s model for the T527/VIP9000 architecture.
* `npu_code/npu_detect.py` - The main Flask web server. It manages the GStreamer subprocess, safely consumes the `fdsink` byte-stream, runs the NPU inference, decodes the YOLO bounding boxes, and serves the UI.
* `scripts/start_all.sh` - The orchestration script. It kills zombie processes, boots MediaMTX, initializes the ISP, and starts the Python server.

## Setup & Installation

### 1. Compile the NPU Wrapper
You must compile the C-wrapper on your Radxa board to generate the `libawnn_npu.so` library that Python uses:
```bash
cd npu_code
bash build.sh
```

### 2. Install Dependencies
```bash
python3 -m venv yolo_env
source yolo_env/bin/activate
pip install -r requirements.txt
```

### 3. Setup MediaMTX
Ensure you have the `mediamtx` binary downloaded and placed in `/home/radxa/` (or update `scripts/start_all.sh` with your correct path).

### 4. Run the Pipeline
```bash
bash scripts/start_all.sh
```

**Access Points:**
- **Raw WebRTC Stream (0% CPU):** `http://<RADXA_IP>:8889/camera`
- **NPU Web UI (Bounding Boxes):** `http://<RADXA_IP>:5000`

## Future Improvements & Roadmap
While this hybrid Python/C architecture completely fixes the crashing bugs and achieves a stable ~15 FPS on the web UI, the Python Global Interpreter Lock (GIL) limits how fast we can push the NPU. 

**Next Steps for Maximum Performance:**
1. **Full C++ Port:** We plan to move the `npu_detect.py` logic entirely into C++ as a native GStreamer plugin (e.g., `awinnsink`). By keeping the entire pipeline in C-space and bypassing Python entirely, the VIP9000 NPU can easily process **40 to 60+ FPS** with sub-10% CPU load.
2. **Thermal Management:** The Allwinner chip reaches 70°C+ during NPU load, causing thermal throttling. Active cooling (heatsink + fan) is strictly required for sustained maximum framerates.
