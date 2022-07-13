# tflite-test
This repo contains scripts and a tool to reproduce the openCL delegate issue with the `reduce_sum`/`Sum` node.
## Building and converting the model
* `model_files` folder contains a very simple model containing a `reduce_sum` node and its corresponding tflite version (FP32).
  * You can also use `generate_dummy_model.py` to build the model and use `convert_model.py` to convert it to tflite.
## tflite_inference tool 
We have implemented a small tool to feed an input to our sample model using `openCL` delegate and display the generated results.
### PREREQUISITES: ###
* Linux host computer
* Connectivity to the target device via adb
* Android NDK, version 22 or later
* CMake 3.18 or later

### BUILD INSTRUCTIONS ###
* Unzip the `tensorflow_lite_cpp_2_9_1_static.zip` file inside the `tflite_inference_tool` folder.
* In a terminal, from `tflite_inference_tool` folder:
```console
$ mkdir build
$ cd build
$ cmake -G "Unix Makefiles"
        -DCMAKE_SYSTEM_NAME=Android 
        -DANDROID_ABI=arm64-v8a 
        -DANDROID_STL=c++_shared 
        -DANDROID_NATIVE_API_LEVEL=27 
        -DCMAKE_VERBOSE_MAKEFILE=ON 
        -DCMAKE_TOOLCHAIN_FILE=<path-to-ndk>/build/cmake/android.toolchain.cmake 
        -DCMAKE_BUILD_TYPE=Release
        -DTensorFlowLite_ROOT=../tensorflow_lite_cpp_2_9_1_static ..
$ make
```
* Here, you must replace <path-to-ndk> with the absolute path of the ndk installed on your computer. If you installed NDK through Android studio, it is typically located at:
    `/home/<username>/Android/Sdk/ndk/<version>/` on Linux

* `tensorflow_lite_cpp_2_9_1_static` is TensorflowFlow Lite library (version 2.9.1) package.
### Run INSTRUCTIONS ###
WARNING: This step will write to your `/data/local/tmp` folder on device. Please make sure existing files in that folder are backed up as needed.

In a terminal, from `tflite_inference_tool` folder:
```console
$ ./run_me.sh
```

The output should be something like this:
```console
INFO: Created TensorFlow Lite delegate for GPU.
INFO: Initialized TensorFlow Lite runtime.
INFO: Replacing 1 node(s) with delegate (TfLiteGpuDelegateV2) node, yielding 1 partitions.
INFO: Initialized OpenCL-based API.
INFO: Created 1 GPU delegate kernels.
260.5, 60.8125, 65.8125, 64.8125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
```
