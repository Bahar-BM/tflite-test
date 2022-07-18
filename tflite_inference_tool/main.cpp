#include "tensorflow/lite/c/c_api.h"    
#include "tensorflow/lite/delegates/gpu/delegate.h" 
#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>

int main(void) {
    TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
    opts.is_precision_loss_allowed = 1;
    opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    opts.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    opts.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;

    TfLiteDelegate* gpuDelegate = TfLiteGpuDelegateV2Create(&opts);
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

    TfLiteInterpreterOptionsAddDelegate(options, gpuDelegate);

    TfLiteModel* model = TfLiteModelCreateFromFile("./fp32_stack.tflite");
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    std::vector<float> randomInput(1*23);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.f, 1.f); 
    std::generate(randomInput.begin(), randomInput.end(), [&](){return dis(gen);});

    TfLiteInterpreterAllocateTensors(interpreter);
    auto* inputTensor0 = TfLiteInterpreterGetInputTensor(interpreter, 0);
    auto* inputTensor1 = TfLiteInterpreterGetInputTensor(interpreter, 1);

    auto status = TfLiteTensorCopyFromBuffer(inputTensor0, randomInput.data(), randomInput.size()*sizeof(float));
    assert(status == kTfLiteOk);

    status = TfLiteTensorCopyFromBuffer(inputTensor1, randomInput.data(), randomInput.size()*sizeof(float));
    assert(status == kTfLiteOk);

    TfLiteInterpreterInvoke(interpreter);

    std::vector<float> output(23*2);
    auto const* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    status = TfLiteTensorCopyToBuffer(outputTensor, output.data(), output.size()*sizeof(float));
    assert(status == kTfLiteOk);

    for(auto v: output)
        std::cout<<v<<", ";
    std::cout<<std::endl;

    TfLiteInterpreterDelete(interpreter);
    TfLiteGpuDelegateV2Delete(gpuDelegate);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
}
