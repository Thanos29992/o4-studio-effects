#!/bin/bash
# Build the studio-effects daemon.
# Run from the project root: ./src/build.sh
set -e

BIN=$HOME/.local/bin/studio-effects
SRC=$HOME/.local/share/studio-effects/src

g++ -O2 -std=c++17 -Wall -Wextra \
  -I/usr/include/openvino \
  -I$SRC \
  $SRC/studio_effects.cpp \
  -lopenvino -lsndfile -lfftw3f \
  -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
  -lv4l2 -lpthread \
  -o "$BIN"

chmod +x "$BIN"
echo "BUILD_OK"
