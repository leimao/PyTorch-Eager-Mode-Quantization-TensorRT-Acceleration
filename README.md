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
[05/19/2024-19:06:41] [I] === Performance summary ===
[05/19/2024-19:06:41] [I] Throughput: 4760.8 qps
[05/19/2024-19:06:41] [I] Latency: min = 0.193115 ms, max = 1.85248 ms, mean = 0.215958 ms, median = 0.195679 ms, percentile(90%) = 0.197632 ms, percentile(95%) = 0.199219 ms, percentile(99%) = 0.910339 ms
[05/19/2024-19:06:41] [I] Enqueue Time: min = 0.0510254 ms, max = 0.203674 ms, mean = 0.0646378 ms, median = 0.0593262 ms, percentile(90%) = 0.0931091 ms, percentile(95%) = 0.0960083 ms, percentile(99%) = 0.105347 ms
[05/19/2024-19:06:41] [I] H2D Latency: min = 0.00366211 ms, max = 0.0654297 ms, mean = 0.00467193 ms, median = 0.00415039 ms, percentile(90%) = 0.0057373 ms, percentile(95%) = 0.00585938 ms, percentile(99%) = 0.00701904 ms
[05/19/2024-19:06:41] [I] GPU Compute Time: min = 0.186279 ms, max = 1.8432 ms, mean = 0.208177 ms, median = 0.188354 ms, percentile(90%) = 0.189423 ms, percentile(95%) = 0.19043 ms, percentile(99%) = 0.903168 ms
[05/19/2024-19:06:41] [I] D2H Latency: min = 0.00256348 ms, max = 0.0194092 ms, mean = 0.00311522 ms, median = 0.00292969 ms, percentile(90%) = 0.00366211 ms, percentile(95%) = 0.00390625 ms, percentile(99%) = 0.00427246 ms
[05/19/2024-19:06:41] [I] Total Host Walltime: 3.00055 s
[05/19/2024-19:06:41] [I] Total GPU Compute Time: 2.97381 s
```

```bash
$ trtexec --onnx=saved_models/resnet_quantized_cifar10_modified.onnx --saveEngine=saved_models/resnet_cifar10_int8.engine --int8 --separateProfileRun --exportLayerInfo=saved_models/resnet_cifar10_int8_layer_info.json --exportProfile=saved_models/resnet_cifar10_int8_profile.json --verbose &> saved_models/resnet_cifar10_int8_build_log.txt
[05/19/2024-19:09:22] [I] === Performance summary ===
[05/19/2024-19:09:22] [I] Throughput: 5614.67 qps
[05/19/2024-19:09:22] [I] Latency: min = 0.157227 ms, max = 1.13483 ms, mean = 0.183582 ms, median = 0.159729 ms, percentile(90%) = 0.162476 ms, percentile(95%) = 0.352905 ms, percentile(99%) = 0.694092 ms
[05/19/2024-19:09:22] [I] Enqueue Time: min = 0.0622559 ms, max = 0.625732 ms, mean = 0.0809989 ms, median = 0.067627 ms, percentile(90%) = 0.112549 ms, percentile(95%) = 0.118164 ms, percentile(99%) = 0.152832 ms
[05/19/2024-19:09:22] [I] H2D Latency: min = 0.0032959 ms, max = 0.296875 ms, mean = 0.00448601 ms, median = 0.00415039 ms, percentile(90%) = 0.00561523 ms, percentile(95%) = 0.00585938 ms, percentile(99%) = 0.00720215 ms
[05/19/2024-19:09:22] [I] GPU Compute Time: min = 0.150391 ms, max = 1.1264 ms, mean = 0.17584 ms, median = 0.152588 ms, percentile(90%) = 0.153625 ms, percentile(95%) = 0.345093 ms, percentile(99%) = 0.687012 ms
[05/19/2024-19:09:22] [I] D2H Latency: min = 0.00244141 ms, max = 0.0214844 ms, mean = 0.0032584 ms, median = 0.00292969 ms, percentile(90%) = 0.00415039 ms, percentile(95%) = 0.00439453 ms, percentile(99%) = 0.00488281 ms
[05/19/2024-19:09:22] [I] Total Host Walltime: 3.00053 s
[05/19/2024-19:09:22] [I] Total GPU Compute Time: 2.96238 s
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
Evaluation Accuracy: 0.8512
```

The accuracy of the INT8-quantized ResNet TensorRT engine matches the PyTorch INT8-quantized model accuracy, suggesting that the INT8-quantized TensorRT engine was built correctly.

## References

- [PyTorch Static Quantization](https://leimao.github.io/blog/PyTorch-Static-Quantization/)
- [PyTorch Eager Mode Quantization TensorRT Acceleration](https://leimao.github.io/blog/PyTorch-Eager-Mode-Quantization-TensorRT-Acceleration/)
