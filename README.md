# Radxa VIP9000 Hardware-Accelerated YOLOv5 Pipeline.

This repository provides an ultra-optimized, zero-copy architecture for running a live camera stream and YOLOv5s real-time person detection on Radxa boards equipped with the Allwinner ISP and Vivante VIP9000 NPU (e.g., Radxa ZERO 3W, Cubie A7S).

## The Goal
The objective of this project was to achieve **Real-Time NPU Object Detection** (specifically YOLOv5s person detection) while simultaneously streaming a pristine, low-latency 1080p 30FPS camera feed over the web, without melting the CPU.

## The Problem: The Allwinner ISP Deadlock
If you try to build an NPU pipeline on these boards using standard tutorials, you will hit a massive wall. Specifically, the Allwinner ISP (`en-awisp=1`) and GStreamer's memory mapping interface have severe driver-level race conditions. 

**What we faced:**
1. **Kernel Panics & Lockups:** When attempting to capture video via OpenCV (`cv2.VideoCapture`), or when using software color conversion (`videoconvert`) to bridge the camera to the encoder, the pipeline deadlocks. Mutexes lock up (`__lll_lock_wait`), GStreamer throws `GST_IS_BUFFER` assertions, and the entire board freezes requiring a hard power cycle. (This is a [known, currently unresolved race condition in the Radxa community](https://forum.radxa.com/t/a7s-gstreamer-current-status-for-real-time-npu/31163) occurring during buffer-pool negotiation and pipeline teardown).
2. **Color Space Glitches:** Forcing the pipeline through software `videoconvert` triggers NV12/YV12 buffer sizing mismatches, resulting in horrible green and purple tints.
3. **CPU Bottlenecks:** Forcing the CPU to handle video encoding maxes out the processor, severely throttling the NPU inference frame rate.

## The Solution: Native NV12 Bypass & Dual-Branch `fdsink`
We abandoned the broken software conversion and shared-memory mapping approach entirely. We architected a custom, decoupled pipeline that bypasses the ISP driver bugs by relying strictly on native hardware formats.

The key breakthrough is that the Allwinner ISP (`v4l2src en-awisp=1`) natively outputs **NV12** pixel format, and the hardware encoder (`omxh264videoenc`) natively expects **NV12**. By removing the software `videoconvert` step entirely from the encoding branch, we bypassed the exact buffer negotiation crash confirmed by other developers. 

Furthermore, by orchestrating the process teardown with forceful system kills (`pkill -9`) rather than graceful GStreamer `PLAYING -> NULL` state transitions, we made the architecture completely immune to the `gst_pad_stop_task` teardown deadlocks that plague this SoC. **Note:** This is a deliberate architectural workaround, not a library patch. We do not fix the bug; we avoid triggering it entirely. This pipeline was tested and proven stable on the known-vulnerable GStreamer Core Library version **1.18.4**—proving that this architecture actively routes around the core library bugs rather than relying on an upstream fix.

Instead of fighting GStreamer memory leaks in Python, we use a single, rock-solid GStreamer CLI subprocess (`gst-launch-1.0`) to split the hardware feed into two independent branches:

1. **Zero-Copy WebRTC (H.264):** One branch of the `tee` pipes pristine 1080p frames directly into the silicon `omxh264videoenc` hardware encoder. This streams to the web at 0% CPU load with perfect colors.
2. **Decoupled `fdsink` Capture:** The second branch converts the frames to flat `BGR` bytes and pipes them to `stdout` via `fdsink`. Python simply reads this byte-stream safely from standard output. Because we aren't mapping shared memory buffers, this eliminates the shared-memory buffer negotiation path where the deadlock occurs.

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

## The C++ Native Port (Maximum Performance & Zero Lag)
While the hybrid Python/C architecture completely fixes the crashing bugs, the Python Global Interpreter Lock (GIL) and stdout byte-reading overhead inherently limit how fast we can push the NPU. 

To solve this, we have written a **fully native, heavily optimized C++ version** (`npu_detect.cpp`).

**Key C++ Optimizations Included:**
1. **Hardware Encoder Sudo Privileges:** The Allwinner hardware video engine (`/dev/cedar_dev`) strictly requires `root` privileges to allocate memory buffers. If run as a standard user, `omxh264videoenc` will instantly crash with `Failed to open encoder`. Our C++ launch script safely wraps the execution with elevated privileges to guarantee hardware encoder initialization.
2. **WebRTC Latency Fix:** Standard GStreamer `queue` elements buffer frames indefinitely if the hardware encoder falls slightly behind, resulting in up to 2 seconds of massive WebRTC delay. We re-engineered the pipeline to use `leaky=downstream max-size-buffers=2` queues on the WebRTC branch, ensuring the live stream strictly drops late frames and remains 100% real-time.
3. **OpenCV Hardware Vectorization:** Transforming the 1080p camera frames from HWC (Interleaved) to CHW (Planar) format for the NPU requires shifting 1.2 million pixels. Using a standard nested C++ `for` loop destroyed the CPU memory bandwidth. We completely replaced this with OpenCV's highly optimized, hardware-vectorized `cv::split` function, drastically reducing CPU overhead.
4. **Dual-Resolution CPU Offloading:** While the WebRTC branch maintains a zero-copy 1080p stream, the NPU branch uses GStreamer's `videoscale` to natively downscale the camera feed to `640x360` (perfect 16:9, 4-byte memory stride aligned). This drops the OpenCV pixel conversion and JPEG encoding payload by 85%, completely relaxing the Radxa's Big Cores (Cores 6 & 7) and preventing thermal throttling.

**How to use the C++ Engine:**
1. Build it:
   ```bash
   cd npu_code
   bash build_cpp.sh
   ```
2. Run it (the script will automatically handle the `sudo` password to unlock the hardware encoder):
   ```bash
   bash scripts/start_all_cpp.sh
   ```
This architecture maxes out the physical mathematical limits of the board. The NPU crunches YOLO matrices at an incredible ~39ms (25+ FPS), while the CPU concurrently maxes out at ~19 FPS running the GStreamer `videoconvert` and JPEG compressions.

## Thermal Management & The 416 MHz Kernel Bug
The Allwinner A733/T527 chip reaches 65°C+ during NPU load, which triggers aggressive thermal throttling. **Active cooling (heatsink + fan) is strictly required.**

**CRITICAL KERNEL BUG:** If your board hits 65°C, the Linux kernel's thermal daemon will panic and forcefully rewrite the CPU frequency limit (`scaling_max_freq`) to its lowest state: **416 MHz**. 
The kernel *will not* revert this when the chip cools down! Even if you attach a fan later, your board will be permanently crippled at 416 MHz, causing massive framerate drops in the pipeline. To fix this, you must physically attach a 5V fan to the SoC and reboot the board entirely to clear the locked thermal panic state and restore the full 1.8 GHz clock speed.
