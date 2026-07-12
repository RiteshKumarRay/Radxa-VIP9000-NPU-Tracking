#!/bin/bash
# Build script for the AWNN NPU shim on the Radxa board

echo "Compiling libawnn_npu.so..."
gcc -shared -fPIC awnn_shim.c -o libawnn_npu.so -lNBGlinker -lVIPhal -I. -Wall

if [ $? -eq 0 ]; then
    echo "Successfully built libawnn_npu.so"
else
    echo "Build failed. Make sure libNBGlinker.so and libVIPhal.so are in your LD_LIBRARY_PATH or /usr/lib."
fi
