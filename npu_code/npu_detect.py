import sys
import ctypes
import time
import threading
import subprocess
import cv2
import numpy as np
from flask import Flask, Response, render_template_string

# ── Paths ─────────────────────────────────────────────────────────────
NPU_LIB  = "/home/radxa/npu/libawnn_npu.so"
MODEL_NB = "/home/radxa/npu/yolov5s.nb"

INPUT_H = 640
INPUT_W = 640
FRAME_W = 1920
FRAME_H = 1080
FRAME_SIZE = FRAME_W * FRAME_H * 3

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
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
det_lock = threading.Lock()
running = True

app = Flask(__name__)

# ── Sigmoid ───────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

# ── YOLOv5 decoder ────────────────────────────────────────────────────
def decode_yolov5(raw_bufs, out_elems, orig_h, orig_w,
                  conf_thresh=0.25, iou_thresh=0.45):
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

        gy, gx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')

        for a_i, (aw, ah) in enumerate(anchors):
            pred = arr[a_i]

            obj  = pred[..., 4]
            cls  = pred[..., 5:]
            person_conf = obj * cls[..., PERSON_IDX]

            mask = person_conf > conf_thresh
            if not mask.any():
                continue

            px = (pred[mask, 0] * 2 - 0.5 + gx[mask]) * stride
            py = (pred[mask, 1] * 2 - 0.5 + gy[mask]) * stride
            pw = (pred[mask, 2] * 2) ** 2 * aw
            ph = (pred[mask, 3] * 2) ** 2 * ah

            scale = min(INPUT_W / orig_w, INPUT_H / orig_h)
            dw = (INPUT_W - orig_w * scale) / 2
            dh = (INPUT_H - orig_h * scale) / 2
            
            px_unpad = px - dw
            py_unpad = py - dh
            
            x1 = (px_unpad - pw / 2) / scale
            y1 = (py_unpad - ph / 2) / scale
            x2 = (px_unpad + pw / 2) / scale
            y2 = (py_unpad + ph / 2) / scale

            scores = person_conf[mask]
            for j in range(len(scores)):
                all_boxes.append([float(x1[j]), float(y1[j]),
                                  float(x2[j] - x1[j]), float(y2[j] - y1[j])])
                all_scores.append(float(scores[j]))

    if not all_boxes:
        return []

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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 3)
        label = f"Person {score:.2f}"
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 10, y1), (0, 200, 60), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return frame

# ── NPU inference thread ──────────────────────────────────────────────
def npu_inference_thread():
    global latest_detections, running

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

    while running:
        with frame_lock:
            frame = latest_frame
            
        if frame is None:
            time.sleep(0.01)
            continue

        orig_h, orig_w = FRAME_H, FRAME_W

        # Preprocess: Letterbox → RGB → uint8 (NCHW)
        scale = min(INPUT_W / orig_w, INPUT_H / orig_h)
        nw, nh = int(orig_w * scale), int(orig_h * scale)
        img_resized = cv2.resize(frame, (nw, nh))
        
        img_pad = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)
        dw = (INPUT_W - nw) // 2
        dh = (INPUT_H - nh) // 2
        img_pad[dh:dh+nh, dw:dw+nw, :] = img_resized

        # FIX GREEN TINT for the NPU explicitly since we get BGR
        img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
        
        inp = img_rgb.transpose(2, 0, 1).astype(np.uint8).flatten()
        buf = inp.ctypes.data_as(ctypes.c_void_p)
        bufs = (ctypes.c_void_p * 1)(buf)

        t0 = time.time()
        lib.awnn_set_input_buffers(ctx, bufs)
        lib.awnn_run(ctx)
        raw_out = lib.awnn_get_output_buffers(ctx)
        inf_ms = int((time.time() - t0) * 1000)

        dets = decode_yolov5(raw_out, out_elems, orig_h, orig_w)

        with det_lock:
            latest_detections = dets

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 5.0:
            fps = frame_count / elapsed
            print(f"[NPU] FPS={fps:.1f}  inference={inf_ms}ms  persons={len(dets)}", flush=True)
            frame_count = 0
            fps_timer   = time.time()

def read_exact(pipe, size):
    buf = bytearray()
    while len(buf) < size:
        chunk = pipe.read(size - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

def generate_mjpeg():
    while running:
        with frame_lock:
            frame_out = latest_frame.copy() if latest_frame is not None else None
        
        with det_lock:
            dets = list(latest_detections)
            
        if frame_out is not None:
            frame_out = draw_detections(frame_out, dets)
            
            ret, jpeg = cv2.imencode('.jpg', frame_out)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template_string("""
    <html>
      <head>
        <title>Radxa NPU Stream</title>
        <style>
          body { background-color: #111; color: white; text-align: center; font-family: sans-serif; }
          img { border: 2px solid #444; border-radius: 8px; max-width: 90%; margin-top: 20px; }
        </style>
      </head>
      <body>
        <h1>NPU Person Detection (Live)</h1>
        <img src="/video_feed" />
      </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

def flask_thread():
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)

def main():
    global latest_frame, running

    print("[MAIN] Starting GStreamer hardware encoder + stdout pipe...", flush=True)
    
    gst_cmd = [
        "gst-launch-1.0", "-q",
        "v4l2src", "device=/dev/video0", "en-awisp=1", "en-largemode=0", "do-timestamp=true", "!",
        "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1", "!",
        "tee", "name=t",
        # WebRTC Hardware stream (Zero-Copy)
        "t.", "!", "queue", "max-size-buffers=60", "!", "omxh264videoenc", "target-bitrate=8000000", "!", "h264parse", "config-interval=1", "!", "rtspclientsink", "location=rtsp://127.0.0.1:8554/camera",
        # OpenCV Python stdout stream
        "t.", "!", "queue", "leaky=1", "max-size-buffers=2", "!", "videoconvert", "!", "video/x-raw,format=BGR", "!", "fdsink"
    ]
    
    p = subprocess.Popen(gst_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    t = threading.Thread(target=npu_inference_thread)
    t.start()
    
    ft = threading.Thread(target=flask_thread)
    ft.daemon = True
    ft.start()

    print("[MAIN] Pipeline fully running! Reading from stdout...", flush=True)
    try:
        while True:
            raw_frame = read_exact(p.stdout, FRAME_SIZE)
            if not raw_frame:
                print("[MAIN] Camera read failed or EOF")
                break
                
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((FRAME_H, FRAME_W, 3))
            
            with frame_lock:
                latest_frame = frame
                
    except KeyboardInterrupt:
        print("[MAIN] Interrupted by user.")
    finally:
        running = False
        p.terminate()

if __name__ == "__main__":
    main()
