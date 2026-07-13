#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <cmath>
#include <chrono>
#include <string>
#include <algorithm>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Web Server
#include "httplib.h"

using namespace std;
using namespace cv;

// ── NPU C-API Shim Declarations ──────────────────────────────────────────────
extern "C" {
    void awnn_init();
    void awnn_uninit();
    void* awnn_create(const char* model_path);
    void awnn_destroy(void* ctx);
    void awnn_set_input_buffers(void* ctx, void** buffers);
    void awnn_run(void* ctx);
    float** awnn_get_output_buffers(void* ctx);
    unsigned int awnn_get_output_count(void* ctx);
    unsigned int awnn_get_output_elements(void* ctx, int index);
}

// ── Constants ────────────────────────────────────────────────────────────────
const char* NPU_MODEL = "/home/radxa/npu/yolov5s.nb";

const int INPUT_H = 640;
const int INPUT_W = 640;
const int FRAME_W = 1920;
const int FRAME_H = 1080;

const int NC = 80;
const int PERSON_IDX = 0;
const float CONF_THRESH = 0.25f;
const float IOU_THRESH = 0.45f;

const vector<vector<pair<float, float>>> ANCHORS = {
    {{10, 13}, {16, 30}, {33, 23}},      // stride 8
    {{30, 61}, {62, 45}, {59, 119}},     // stride 16
    {{116, 90}, {156, 198}, {373, 326}}  // stride 32
};
const vector<int> STRIDES = {8, 16, 32};

// ── Global State ─────────────────────────────────────────────────────────────
atomic<bool> g_running(true);

mutex g_frame_mutex;
Mat g_latest_frame;

struct Detection {
    Rect bbox;
    float score;
};

mutex g_det_mutex;
vector<Detection> g_latest_detections;

// ── YOLOv5 Decoder ───────────────────────────────────────────────────────────
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(std::max(std::min(-x, 88.0f), -88.0f)));
}

vector<Detection> decode_yolov5(float** raw_bufs, const vector<unsigned int>& out_elems, int orig_h, int orig_w) {
    vector<Rect2d> boxes;
    vector<float> scores;

    float scale = min((float)INPUT_W / orig_w, (float)INPUT_H / orig_h);
    float dw = (INPUT_W - orig_w * scale) / 2.0f;
    float dh = (INPUT_H - orig_h * scale) / 2.0f;

    for (size_t head_i = 0; head_i < STRIDES.size(); ++head_i) {
        int stride = STRIDES[head_i];
        int grid_h = INPUT_H / stride;
        int grid_w = INPUT_W / stride;
        unsigned int expected = 3 * grid_h * grid_w * (5 + NC);

        if (out_elems[head_i] != expected) continue;

        float* arr = raw_bufs[head_i];
        int b_idx = 0;

        for (int a_i = 0; a_i < 3; ++a_i) {
            float aw = ANCHORS[head_i][a_i].first;
            float ah = ANCHORS[head_i][a_i].second;

            for (int gy = 0; gy < grid_h; ++gy) {
                for (int gx = 0; gx < grid_w; ++gx) {
                    float* pred = &arr[b_idx];
                    b_idx += (5 + NC);

                    float obj = sigmoid(pred[4]);
                    float cls_person = sigmoid(pred[5 + PERSON_IDX]);
                    float conf = obj * cls_person;

                    if (conf > CONF_THRESH) {
                        float px = (sigmoid(pred[0]) * 2.0f - 0.5f + gx) * stride;
                        float py = (sigmoid(pred[1]) * 2.0f - 0.5f + gy) * stride;
                        float pw = pow(sigmoid(pred[2]) * 2.0f, 2) * aw;
                        float ph = pow(sigmoid(pred[3]) * 2.0f, 2) * ah;

                        float px_unpad = px - dw;
                        float py_unpad = py - dh;

                        float x1 = (px_unpad - pw / 2.0f) / scale;
                        float y1 = (py_unpad - ph / 2.0f) / scale;
                        float x2 = (px_unpad + pw / 2.0f) / scale;
                        float y2 = (py_unpad + ph / 2.0f) / scale;

                        boxes.push_back(Rect2d(x1, y1, x2 - x1, y2 - y1));
                        scores.push_back(conf);
                    }
                }
            }
        }
    }

    vector<int> indices;
    dnn::NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH, indices);

    vector<Detection> results;
    for (int idx : indices) {
        Detection d;
        d.bbox = Rect(boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height);
        d.score = scores[idx];
        results.push_back(d);
    }
    return results;
}

// ── NPU Inference Thread ─────────────────────────────────────────────────────
void npu_inference_thread() {
    cout << "[NPU] Initializing VIP9000 NPU (Native C++)..." << endl;
    awnn_init();

    cout << "[NPU] Loading model: " << NPU_MODEL << endl;
    void* ctx = awnn_create(NPU_MODEL);
    if (!ctx) {
        cerr << "[NPU] FAILED to load model — exiting" << endl;
        g_running = false;
        return;
    }

    int n_out = awnn_get_output_count(ctx);
    vector<unsigned int> out_elems;
    for (int i = 0; i < n_out; ++i) {
        out_elems.push_back(awnn_get_output_elements(ctx, i));
    }
    cout << "[NPU] Model ready: " << n_out << " outputs" << endl;

    int frame_count = 0;
    auto fps_timer = chrono::steady_clock::now();

    while (g_running) {
        Mat frame;
        {
            lock_guard<mutex> lock(g_frame_mutex);
            if (!g_latest_frame.empty()) {
                frame = g_latest_frame.clone(); // Shallow clone of reference, safe
            }
        }

        if (frame.empty()) {
            this_thread::sleep_for(chrono::milliseconds(5));
            continue;
        }

        int orig_h = frame.rows;
        int orig_w = frame.cols;

        // Preprocess: Letterbox -> RGB
        float scale = min((float)INPUT_W / orig_w, (float)INPUT_H / orig_h);
        int nw = orig_w * scale;
        int nh = orig_h * scale;

        Mat resized;
        cv::resize(frame, resized, Size(nw, nh));

        Mat img_pad(INPUT_H, INPUT_W, CV_8UC3, Scalar(114, 114, 114));
        int dw = (INPUT_W - nw) / 2;
        int dh = (INPUT_H - nh) / 2;
        resized.copyTo(img_pad(Rect(dw, dh, nw, nh)));

        Mat img_rgb;
        cv::cvtColor(img_pad, img_rgb, COLOR_BGR2RGB);

        // HWC to CHW (NCHW) - Optimized using OpenCV vectorized split
        vector<Mat> channels;
        cv::split(img_rgb, channels);
        
        vector<uint8_t> chw_data(INPUT_H * INPUT_W * 3);
        int stride = INPUT_H * INPUT_W;
        memcpy(chw_data.data(), channels[0].data, stride);
        memcpy(chw_data.data() + stride, channels[1].data, stride);
        memcpy(chw_data.data() + 2 * stride, channels[2].data, stride);
        
        void* buf_ptr = chw_data.data();
        void** bufs = &buf_ptr;

        auto t0 = chrono::high_resolution_clock::now();
        
        awnn_set_input_buffers(ctx, bufs);
        awnn_run(ctx);
        float** raw_out = awnn_get_output_buffers(ctx);
        
        auto t1 = chrono::high_resolution_clock::now();
        int inf_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

        vector<Detection> dets = decode_yolov5(raw_out, out_elems, orig_h, orig_w);

        {
            lock_guard<mutex> lock(g_det_mutex);
            g_latest_detections = dets;
        }

        frame_count++;
        auto elapsed = chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - fps_timer).count();
        if (elapsed >= 5) {
            float fps = (float)frame_count / elapsed;
            cout << "[NPU] FPS=" << fps << "  inference=" << inf_ms << "ms  persons=" << dets.size() << endl;
            frame_count = 0;
            fps_timer = chrono::steady_clock::now();
        }
    }

    awnn_destroy(ctx);
    awnn_uninit();
}

// ── Web Server ───────────────────────────────────────────────────────────────
void web_server_thread() {
    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(
            "<html><head><title>Radxa NPU Stream (C++)</title>"
            "<style>body { background-color: #111; color: white; text-align: center; font-family: sans-serif; }"
            "img { border: 2px solid #444; border-radius: 8px; max-width: 90%; margin-top: 20px; }</style>"
            "</head><body><h1>NPU Person Detection (C++ Native)</h1><img src=\"/video_feed\" /></body></html>",
            "text/html"
        );
    });

    svr.Get("/video_feed", [](const httplib::Request&, httplib::Response& res) {
        res.set_chunked_content_provider(
            "multipart/x-mixed-replace; boundary=frame",
            [](size_t offset, httplib::DataSink& sink) {
                while (g_running) {
                    Mat frame;
                    vector<Detection> dets;

                    {
                        lock_guard<mutex> lock(g_frame_mutex);
                        if (!g_latest_frame.empty()) frame = g_latest_frame.clone();
                    }
                    {
                        lock_guard<mutex> lock(g_det_mutex);
                        dets = g_latest_detections;
                    }

                    if (!frame.empty()) {
                        for (const auto& d : dets) {
                            cv::rectangle(frame, d.bbox, Scalar(0, 255, 80), 3);
                            string label = "Person " + to_string(d.score).substr(0,4);
                            int baseline = 0;
                            Size sz = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
                            cv::rectangle(frame, Point(d.bbox.x, d.bbox.y - sz.height - 10), Point(d.bbox.x + sz.width + 10, d.bbox.y), Scalar(0, 200, 60), -1);
                            cv::putText(frame, label, Point(d.bbox.x + 5, d.bbox.y - 5), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, LINE_AA);
                        }

                        vector<uchar> buf;
                        cv::imencode(".jpg", frame, buf);

                        string header = "--frame\r\nContent-Type: image/jpeg\r\n\r\n";
                        sink.write(header.data(), header.size());
                        sink.write((const char*)buf.data(), buf.size());
                        sink.write("\r\n\r\n", 4);
                    }
                    this_thread::sleep_for(chrono::milliseconds(33)); // ~30 FPS limit for MJPEG
                }
                return false; 
            }
        );
    });

    cout << "[WEB] Starting C++ HTTP Server on 0.0.0.0:5000" << endl;
    svr.listen("0.0.0.0", 5000);
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    string pipeline_str = 
        "gst-launch-1.0 -q "
        "v4l2src device=/dev/video0 en-awisp=1 en-largemode=0 do-timestamp=true ! "
        "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! tee name=t "
        "t. ! queue leaky=1 max-size-buffers=2 ! omxh264videoenc target-bitrate=8000000 ! h264parse config-interval=1 ! rtspclientsink location=rtsp://127.0.0.1:8554/camera "
        "t. ! queue leaky=1 max-size-buffers=2 ! videoconvert ! video/x-raw,format=BGR ! fdsink";

    cout << "[MAIN] Launching GStreamer C++ Pipeline via popen()..." << endl;
    FILE* pipe = popen(pipeline_str.c_str(), "r");
    if (!pipe) {
        cerr << "Failed to start GStreamer pipeline!" << endl;
        return -1;
    }

    thread npu_thread(npu_inference_thread);
    npu_thread.detach();
    
    thread web_thread(web_server_thread);
    web_thread.detach();

    cout << "[MAIN] All threads running. Reading from stdout..." << endl;
    
    size_t frame_size = FRAME_W * FRAME_H * 3;
    vector<uint8_t> buffer(frame_size);

    while (g_running) {
        size_t bytes_read = fread(buffer.data(), 1, frame_size, pipe);
        if (bytes_read != frame_size) {
            cerr << "[MAIN] Camera stream ended or failed to read full frame." << endl;
            break;
        }

        Mat frame(FRAME_H, FRAME_W, CV_8UC3, buffer.data());
        {
            lock_guard<mutex> lock(g_frame_mutex);
            frame.copyTo(g_latest_frame);
        }
    }

    cout << "[MAIN] Shutting down..." << endl;
    g_running = false;
    pclose(pipe);

    return 0;
}
