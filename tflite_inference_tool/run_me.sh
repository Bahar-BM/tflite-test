#!/bin/bash

adb push ./build/model_test /data/local/tmp

adb push ./model_files/fp32_stack.tflite /data/local/tmp

adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./model_test"
