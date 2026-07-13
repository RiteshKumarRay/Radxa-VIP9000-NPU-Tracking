#!/bin/bash
# Build script for the native C++ NPU application

echo "=============================================="
echo "  Building C++ NPU Native Pipeline"
echo "=============================================="

# Ensure cpp-httplib is present
if [ ! -f "httplib.h" ]; then
    echo "Downloading cpp-httplib dependency..."
    wget -q https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h -O httplib.h
fi

# Compile the C shim separately as C code
echo "Compiling C shim..."
gcc -c -fPIC awnn_shim.c -o awnn_shim.o -I.

# Compile the C++ file and link
echo "Compiling C++ engine..."
g++ -O3 npu_detect.cpp awnn_shim.o -o npu_detect_cpp \
    -I. \
    $(pkg-config --cflags --libs opencv4) \
    -lNBGlinker -lVIPhal -lpthread

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully built 'npu_detect_cpp'"
    echo "To run:"
    echo "  export LD_LIBRARY_PATH=/home/radxa/npu"
    echo "  ./npu_detect_cpp"
else
    echo "❌ Build failed. Please ensure GStreamer and OpenCV dev packages are installed."
fi
