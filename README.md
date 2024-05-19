# PyTorch Eager Mode Quantization TensorRT Acceleration

## Introduction

PyTorch quantization models from the native PyTorch eager model quantization APIs are not natively compatible with TensorRT. This repository demonstrates how to quantize a PyTorch ResNet model using eager mode quantization and then convert the quantized PyTorch model to a TensorRT engine for acceleration.

## Usages

### Build Docker Image

To build the custom Docker image, run the following command.

```bash
$ docker build -f docker/pytorch-tensorrt.Dockerfile --no-cache --tag=pytorch-tensorrt:2.3.0 .
```

### Run Docker Container

To run the Docker container, run the following command.

```bash
$ docker run -it --rm --gpus device=0 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/mnt pytorch-tensorrt:2.3.0
```

### ResNet CIFAR10 FP32 Training and INT8 Static Quantization Calibration

A ResNet model will be trained on CIFAR10 dataset using PyTorch and then quantized to INT8 using static quantization using PyTorch eager mode quantization. Per-channel symmetric quantization and per-tensor symmetric quantization will be used for quantizing weights and activations to accommodate [TensorRT INT8 quantization requirements](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-quantization) respectively.

The floating-point and INT8-quantized ResNet models will be exported to ONNX format for TensorRT engine building.

```bash
$ python resnet_torch.py
FP32 Evaluation Accuracy: 0.854
INT8 Evaluation Accuracy: 0.852
FP32 CPU Inference Latency: 2.64 ms / sample
FP32 CUDA Inference Latency: 1.74 ms / sample
INT8 CPU Inference Latency: 46.98 ms / sample
INT8 JIT CPU Inference Latency: 44.84 ms / sample
```

The INT8-quantized ResNet model has almost the same accuracy as the floating-point ResNet model. Probably because PyTorch CPU inference does not support per-channel symmetric quantization well, the INT8-quantized ResNet model CPU inference is much slower comparing to [other quantization schemes, which is not compatible to TensorRT quantization requirements, on the same model](https://leimao.github.io/blog/PyTorch-Static-Quantization/).

### INT8-quantized ResNet ONNX Model Graph Surgery

The exported INT8-quantized ResNet ONNX model has some "bugs" and are not natively compatible with TensorRT. The model graph will be modified to make it compatible with TensorRT.

```bash
$ python onnx_gs.py
```

### Build and Profile TensorRT Engine

The floating-point and INT8-quantized ResNet ONNX models will be built to TensorRT engines and profiled using TensorRT `trtexec` tool on an NVIDIA RTX 3090 GPU.

```bash
$ trtexec --onnx=saved_models/resnet_cifar10.onnx --saveEngine=saved_models/resnet_cifar10_fp16.engine --fp16 --separateProfileRun --exportLayerInfo=saved_models/resnet_cifar10_fp16_layer_info.json --exportProfile=saved_models/resnet_cifar10_fp16_profile.json --verbose &> saved_models/resnet_cifar10_fp16_build_log.txt
[05/19/2024-17:33:07] [I] === Performance summary ===
[05/19/2024-17:33:07] [I] Throughput: 5190.48 qps
[05/19/2024-17:33:07] [I] Latency: min = 0.192139 ms, max = 0.948242 ms, mean = 0.198548 ms, median = 0.19458 ms, percentile(90%) = 0.196777 ms, percentile(95%) = 0.198181 ms, percentile(99%) = 0.219116 ms
[05/19/2024-17:33:07] [I] Enqueue Time: min = 0.0493164 ms, max = 0.168213 ms, mean = 0.0594481 ms, median = 0.0557861 ms, percentile(90%) = 0.0664062 ms, percentile(95%) = 0.090332 ms, percentile(99%) = 0.097168 ms
[05/19/2024-17:33:07] [I] H2D Latency: min = 0.00366211 ms, max = 0.0292969 ms, mean = 0.00444743 ms, median = 0.00411987 ms, percentile(90%) = 0.0057373 ms, percentile(95%) = 0.00585938 ms, percentile(99%) = 0.00610352 ms
[05/19/2024-17:33:07] [I] GPU Compute Time: min = 0.185303 ms, max = 0.941162 ms, mean = 0.190965 ms, median = 0.187378 ms, percentile(90%) = 0.188477 ms, percentile(95%) = 0.189453 ms, percentile(99%) = 0.208984 ms
[05/19/2024-17:33:07] [I] D2H Latency: min = 0.00244141 ms, max = 0.0266113 ms, mean = 0.00313933 ms, median = 0.00292969 ms, percentile(90%) = 0.0038147 ms, percentile(95%) = 0.00402832 ms, percentile(99%) = 0.00439453 ms
[05/19/2024-17:33:07] [I] Total Host Walltime: 3.00049 s
[05/19/2024-17:33:07] [I] Total GPU Compute Time: 2.9741 s
```

```bash
$ trtexec --onnx=saved_models/resnet_quantized_cifar10_modified.onnx --saveEngine=saved_models/resnet_cifar10_int8.engine --int8 --separateProfileRun --exportLayerInfo=saved_models/resnet_cifar10_int8_layer_info.json --exportProfile=saved_models/resnet_cifar10_int8_profile.json --verbose &> saved_models/resnet_cifar10_int8_build_log.txt
[05/19/2024-17:35:46] [I] === Performance summary ===
[05/19/2024-17:35:46] [I] Throughput: 6288.12 qps
[05/19/2024-17:35:46] [I] Latency: min = 0.158447 ms, max = 1.18469 ms, mean = 0.164444 ms, median = 0.160645 ms, percentile(90%) = 0.162354 ms, percentile(95%) = 0.164307 ms, percentile(99%) = 0.22522 ms
[05/19/2024-17:35:46] [I] Enqueue Time: min = 0.0617676 ms, max = 0.507812 ms, mean = 0.0791843 ms, median = 0.0661621 ms, percentile(90%) = 0.108398 ms, percentile(95%) = 0.109619 ms, percentile(99%) = 0.141724 ms
[05/19/2024-17:35:46] [I] H2D Latency: min = 0.00341797 ms, max = 0.210815 ms, mean = 0.00430628 ms, median = 0.00402832 ms, percentile(90%) = 0.00491333 ms, percentile(95%) = 0.0057373 ms, percentile(99%) = 0.0067749 ms
[05/19/2024-17:35:46] [I] GPU Compute Time: min = 0.151489 ms, max = 1.17554 ms, mean = 0.156959 ms, median = 0.153564 ms, percentile(90%) = 0.154602 ms, percentile(95%) = 0.154785 ms, percentile(99%) = 0.202637 ms
[05/19/2024-17:35:46] [I] D2H Latency: min = 0.00256348 ms, max = 0.0269775 ms, mean = 0.00318154 ms, median = 0.00292969 ms, percentile(90%) = 0.00378418 ms, percentile(95%) = 0.00408936 ms, percentile(99%) = 0.0045166 ms
[05/19/2024-17:35:46] [I] Total Host Walltime: 3.00106 s
[05/19/2024-17:35:46] [I] Total GPU Compute Time: 2.96197 s
```

Even if the input images to the ResNet model are small (32 x 32) and the batch size is only 1, comparing to the floating-point ResNet engine, the INT8-quantized ResNet engine has a 1.2x latency improvement. The models that have higher math utilization will have more significant latency improvements when quantized to INT8.


### Validate TensorRT Engine

The correctness of the INT8-quantized ResNet TensorRT engine will be validated using a custom TensorRT inference Python script. If the TensorRT engine was built correctly, the accuracy of the INT8-quantized ResNet engine should match the PyTorch INT8-quantized model accuracy.

```bash
$ python resnet_tensorrt.py
Input Tensor:
Tensor Name: x.1 Shape: (1, 3, 32, 32) Data Type: float32 Data Format: TensorFormat.LINEAR
Output Tensor:
Tensor Name: 755 Shape: (1, 10) Data Type: float32 Data Format: TensorFormat.LINEAR
Evaluation Accuracy: 0.8523
```

The accuracy of the INT8-quantized ResNet TensorRT engine matches the PyTorch INT8-quantized model accuracy, suggesting that the INT8-quantized TensorRT engine was built correctly.

## References


* []()



