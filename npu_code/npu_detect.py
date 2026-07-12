import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

"""
npu_detect.py — Person detection using YOLOv5s on the Vivante VIP9000 NPU
via libawnn_npu.so (wraps libNBGlinker + libVIPhal).
Serves annotated MJPEG video over Flask at port 5000.
"""
import sys
import ctypes
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response

# ── Paths ─────────────────────────────────────────────────────────────
NPU_LIB  = "/home/radxa/npu/libawnn_npu.so"
MODEL_NB = "/home/radxa/npu/yolov5s.nb"
RTSP_URL = "rtsp://127.0.0.1:8554/camera"

INPUT_H = 640
INPUT_W = 640

# YOLOv5 COCO anchor boxes for 640-input (P3/P4/P5)
ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],      # small (80×80)
    [(30, 61), (62, 45), (59, 119)],     # medium (40×40)
    [(116, 90), (156, 198), (373, 326)], # large (20×20)
]
STRIDES = [8, 16, 32]
NC = 80   # COCO classes
PERSON_IDX = 0

# ── Load shared library ───────────────────────────────────────────────
lib = ctypes.CDLL(NPU_LIB, use_errno=True)
lib.awnn_init.restype    = None
lib.awnn_uninit.restype  = None
lib.awnn_create.argtypes = [ctypes.c_char_p]
lib.awnn_create.restype  = ctypes.c_void_p
lib.awnn_destroy.argtypes = [ctypes.c_void_p]
lib.awnn_destroy.restype  = None
lib.awnn_set_input_buffers.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
lib.awnn_set_input_buffers.restype  = None
lib.awnn_run.argtypes = [ctypes.c_void_p]
lib.awnn_run.restype  = None
lib.awnn_get_output_buffers.argtypes = [ctypes.c_void_p]
lib.awnn_get_output_buffers.restype  = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
lib.awnn_get_output_count.argtypes   = [ctypes.c_void_p]
lib.awnn_get_output_count.restype    = ctypes.c_uint32
lib.awnn_get_output_elements.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.awnn_get_output_elements.restype  = ctypes.c_uint32

# ── Shared state ──────────────────────────────────────────────────────
raw_frame   = None
latest_jpeg = None
lock        = threading.Lock()
frame_ready = threading.Event()

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NPU Person Detection</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:#0a0a0f; color:#e0e0ff;
           font-family:'Segoe UI',sans-serif;
           display:flex; flex-direction:column; align-items:center;
           min-height:100vh; padding:20px; }
    h1 { font-size:1.6rem; font-weight:600; margin-bottom:12px;
         background:linear-gradient(90deg,#7c3aed,#2563eb);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .badge { background:#1e1b4b; border:1px solid #4c1d95; border-radius:20px;
             padding:4px 14px; font-size:0.75rem; color:#a5b4fc; margin-bottom:20px; }
    .frame { border:2px solid #312e81; border-radius:12px; overflow:hidden;
             box-shadow:0 0 40px rgba(124,58,237,0.3); max-width:100%; }
    .frame img { display:block; width:100%; max-width:960px; height:auto; }
    .footer { margin-top:16px; font-size:0.7rem; color:#6366f1; opacity:0.6; }
  </style>
</head>
<body>
  <h1>🎯 VIP9000 NPU · Person Detection</h1>
  <div class="badge">3 TOPS · YOLOv5s · Live</div>
  <div class="frame"><img src="/video_feed" alt="Connecting..."></div>
  <div class="footer">NPU-accelerated inference · Live MJPEG stream · port 5000</div>
</body>
</html>'''

@app.route('/')
def index():
    return HTML

def generate():
    while True:
        if latest_jpeg is None:
            time.sleep(0.05)
            continue
        with lock:
            frame = latest_jpeg
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ── Sigmoid ───────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

# ── YOLOv5 decoder ────────────────────────────────────────────────────
def decode_yolov5(raw_bufs, out_elems, orig_h, orig_w,
                  conf_thresh=0.4, iou_thresh=0.45):
    """
    Decode 3-head YOLOv5 output into person detections.
    Each head shape: (3, Hg, Wg, 5+NC), stride=8/16/32.
    """
    all_boxes  = []
    all_scores = []

    for head_i, (anchors, stride) in enumerate(zip(ANCHORS, STRIDES)):
        n = out_elems[head_i]
        grid_h = INPUT_H // stride
        grid_w = INPUT_W // stride
        expected = 3 * grid_h * grid_w * (5 + NC)
        if n != expected:
            continue

        arr = np.ctypeslib.as_array(raw_bufs[head_i], shape=(n,)).copy()
        arr = arr.reshape(3, grid_h, grid_w, 5 + NC)
        arr = sigmoid(arr)

        # Build grid
        gy, gx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')

        for a_i, (aw, ah) in enumerate(anchors):
            pred = arr[a_i]  # (grid_h, grid_w, 5+NC)

            obj  = pred[..., 4]          # objectness
            cls  = pred[..., 5:]         # class probs
            person_conf = obj * cls[..., PERSON_IDX]  # (Hg, Wg)

            mask = person_conf > conf_thresh
            if not mask.any():
                continue

            # Decode boxes for valid anchors
            px = (pred[mask, 0] * 2 - 0.5 + gx[mask]) * stride  # pixel cx
            py = (pred[mask, 1] * 2 - 0.5 + gy[mask]) * stride  # pixel cy
            pw = (pred[mask, 2] * 2) ** 2 * aw                   # pixel w
            ph = (pred[mask, 3] * 2) ** 2 * ah                   # pixel h

            # Scale to original image
            sx = orig_w / INPUT_W
            sy = orig_h / INPUT_H
            x1 = (px - pw / 2) * sx
            y1 = (py - ph / 2) * sy
            x2 = (px + pw / 2) * sx
            y2 = (py + ph / 2) * sy

            scores = person_conf[mask]
            for j in range(len(scores)):
                all_boxes.append([float(x1[j]), float(y1[j]),
                                  float(x2[j] - x1[j]), float(y2[j] - y1[j])])
                all_scores.append(float(scores[j]))

    if not all_boxes:
        return []

    # NMS
    indices = cv2.dnn.NMSBoxes(
        [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in all_boxes],
        all_scores, conf_thresh, iou_thresh)
    if len(indices) == 0:
        return []
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    else:
        indices = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]

    dets = []
    for idx in indices:
        b = all_boxes[idx]
        x1 = int(b[0]); y1 = int(b[1])
        x2 = int(b[0] + b[2]); y2 = int(b[1] + b[3])
        dets.append({'bbox': (x1, y1, x2, y2), 'score': all_scores[idx]})
    return dets

# ── Draw detections ───────────────────────────────────────────────────
def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        score = d['score']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
        label = f"Person {score:.2f}"
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (0, 200, 60), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

# ── NPU inference thread ──────────────────────────────────────────────
def npu_inference():
    global latest_jpeg, raw_frame

    print("[NPU] Initializing VIP9000 NPU...", flush=True)
    lib.awnn_init()

    print(f"[NPU] Loading model: {MODEL_NB}", flush=True)
    ctx = lib.awnn_create(MODEL_NB.encode())
    if not ctx:
        print("[NPU] FAILED to load model — exiting", flush=True)
        return

    n_out = lib.awnn_get_output_count(ctx)
    out_elems = [lib.awnn_get_output_elements(ctx, i) for i in range(n_out)]
    print(f"[NPU] Model ready: {n_out} outputs, sizes={out_elems}", flush=True)

    frame_count = 0
    fps_timer   = time.time()

    while True:
        frame_ready.wait(timeout=2.0)
        frame_ready.clear()

        with lock:
            frame = raw_frame
        if frame is None:
            continue

        orig_h, orig_w = frame.shape[:2]

        # Preprocess: resize → RGB → uint8 (NCHW)
        img = cv2.resize(frame, (INPUT_W, INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = img.transpose(2, 0, 1).astype(np.uint8).flatten()
        buf = inp.ctypes.data_as(ctypes.c_void_p)
        bufs = (ctypes.c_void_p * 1)(buf)

        # Run NPU
        t0 = time.time()
        lib.awnn_set_input_buffers(ctx, bufs)
        lib.awnn_run(ctx)
        raw_out = lib.awnn_get_output_buffers(ctx)
        t1 = time.time()

        # Decode
        try:
            dets = decode_yolov5(raw_out, out_elems, orig_h, orig_w)
        except Exception as e:
            print(f"[NPU] Decode error: {e}", flush=True)
            dets = []

        # Annotate frame
        annotated = draw_detections(frame.copy(), dets)

        # FPS overlay
        inf_ms = int((t1 - t0) * 1000)
        overlay = f"NPU {inf_ms}ms | {len(dets)} person(s)"
        cv2.rectangle(annotated, (0, 0), (310, 32), (0, 0, 0), -1)
        cv2.putText(annotated, overlay, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 1, cv2.LINE_AA)

        # Log every 5 s
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 5.0:
            fps = frame_count / elapsed
            print(f"[NPU] FPS={fps:.1f}  inference={inf_ms}ms  persons={len(dets)}", flush=True)
            frame_count = 0
            fps_timer   = time.time()

        # JPEG → Flask
        ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ret:
            with lock:
                latest_jpeg = buf.tobytes()

# ── Camera reader thread ──────────────────────────────────────────────
def camera_reader():
    global raw_frame
    print("[CAM] Connecting to RTSP stream...", flush=True)
    cap = cv2.VideoCapture(RTSP_URL)
    while not cap.isOpened():
        print("[CAM] Waiting for stream...", flush=True)
        time.sleep(2)
        cap = cv2.VideoCapture(RTSP_URL)
    print("[CAM] Connected!", flush=True)
    fails = 0
    while True:
        ret, frame = cap.read()
        if ret:
            fails = 0
            with lock:
                raw_frame = frame.copy()
            frame_ready.set()
        else:
            fails += 1
            if fails > 10:
                print("[CAM] Reconnecting...", flush=True)
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(RTSP_URL)
                fails = 0
            else:
                time.sleep(0.05)

# ── Entry point ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("[MAIN] Starting camera reader...", flush=True)
    threading.Thread(target=camera_reader, daemon=True).start()

    print("[MAIN] Starting NPU inference...", flush=True)
    threading.Thread(target=npu_inference, daemon=True).start()

    print("[MAIN] Flask on http://0.0.0.0:5000", flush=True)
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
