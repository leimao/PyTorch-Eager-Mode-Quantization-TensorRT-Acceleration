# PyTorch Eager Mode Quantization TensorRT Acceleration

## Introduction

PyTorch quantization models from the native PyTorch eager model quantization APIs are not natively compatible with TensorRT. This repository demonstrates how to quantize a PyTorch ResNet model using eager mode quantization and then convert the quantized PyTorch model to a TensorRT engine for acceleration.

## Usages

### Build Docker Image

To build the custom Docker image, run the following command.

```bash
$ docker build -f docker/pytorch-tensorrt.Dockerfile --no-cache --tag=pytorch-tensorrt:2.3.1 .
```

### Run Docker Container

To run the Docker container, run the following command.

```bash
$ docker run -it --rm --gpus device=0 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/mnt pytorch-tensorrt:2.3.1
```

### ResNet CIFAR10 FP32 Training and INT8 Static Quantization Calibration

A ResNet model will be trained on CIFAR10 dataset using PyTorch and then quantized to INT8 using static quantization using PyTorch eager mode quantization. Per-channel symmetric quantization and per-tensor symmetric quantization will be used for quantizing weights and activations to accommodate [TensorRT INT8 quantization requirements](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-quantization) respectively.

The floating-point and INT8-quantized ResNet models will be exported to ONNX format for TensorRT engine building.

```bash
$ python resnet_torch.py
FP32 Evaluation Accuracy: 0.854
INT8 Evaluation Accuracy: 0.854
FP32 CPU Inference Latency: 2.67 ms / sample
FP32 CUDA Inference Latency: 1.87 ms / sample
INT8 CPU Inference Latency: 47.28 ms / sample
INT8 JIT CPU Inference Latency: 43.36 ms / sample
```

The INT8-quantized ResNet model has almost the same accuracy as the floating-point ResNet model. Probably because PyTorch CPU inference does not support per-channel symmetric quantization well, the INT8-quantized ResNet model CPU inference is much slower comparing to [other quantization schemes](https://leimao.github.io/blog/PyTorch-Static-Quantization/), which is not compatible to TensorRT quantization requirements, on the same model.

### INT8-quantized ResNet ONNX Model Graph Surgery

The exported INT8-quantized ResNet ONNX model has some "bugs" and are not natively compatible with TensorRT. The model graph will be modified to make it compatible with TensorRT.

```bash
$ python onnx_gs.py
```

### Build and Profile TensorRT Engine

The floating-point and INT8-quantized ResNet ONNX models will be built to TensorRT engines and profiled using TensorRT `trtexec` tool on an NVIDIA RTX 3090 GPU.

```bash
$ trtexec --onnx=saved_models/resnet_cifar10.onnx --saveEngine=saved_models/resnet_cifar10_fp16.engine --fp16 --separateProfileRun --exportLayerInfo=saved_models/resnet_cifar10_fp16_layer_info.json --exportProfile=saved_models/resnet_cifar10_fp16_profile.json --verbose &> saved_models/resnet_cifar10_fp16_build_log.txt
[07/22/2024-01:14:18] [I] === Performance summary ===
[07/22/2024-01:14:18] [I] Throughput: 4755.58 qps
[07/22/2024-01:14:18] [I] Latency: min = 0.194138 ms, max = 1.2041 ms, mean = 0.216214 ms, median = 0.197052 ms, percentile(90%) = 0.199219 ms, percentile(95%) = 0.201416 ms, percentile(99%) = 0.98291 ms
[07/22/2024-01:14:18] [I] Enqueue Time: min = 0.0526733 ms, max = 0.149902 ms, mean = 0.0652547 ms, median = 0.0603027 ms, percentile(90%) = 0.0861816 ms, percentile(95%) = 0.0930176 ms, percentile(99%) = 0.104248 ms
[07/22/2024-01:14:18] [I] H2D Latency: min = 0.00341797 ms, max = 0.0146484 ms, mean = 0.00464704 ms, median = 0.00415039 ms, percentile(90%) = 0.00585938 ms, percentile(95%) = 0.00585938 ms, percentile(99%) = 0.00695801 ms
[07/22/2024-01:14:18] [I] GPU Compute Time: min = 0.187378 ms, max = 1.19604 ms, mean = 0.208473 ms, median = 0.189453 ms, percentile(90%) = 0.190491 ms, percentile(95%) = 0.192505 ms, percentile(99%) = 0.974854 ms
[07/22/2024-01:14:18] [I] D2H Latency: min = 0.00256348 ms, max = 0.0172119 ms, mean = 0.00308884 ms, median = 0.00292969 ms, percentile(90%) = 0.003479 ms, percentile(95%) = 0.00390625 ms, percentile(99%) = 0.00427246 ms
[07/22/2024-01:14:18] [I] Total Host Walltime: 3.00068 s
[07/22/2024-01:14:18] [I] Total GPU Compute Time: 2.9749 s
```

```bash
$ trtexec --onnx=saved_models/resnet_quantized_cifar10_modified.onnx --saveEngine=saved_models/resnet_cifar10_int8.engine --int8 --separateProfileRun --exportLayerInfo=saved_models/resnet_cifar10_int8_layer_info.json --exportProfile=saved_models/resnet_cifar10_int8_profile.json --verbose &> saved_models/resnet_cifar10_int8_build_log.txt
[07/22/2024-02:02:08] [I] === Performance summary ===
[07/22/2024-02:02:08] [I] Throughput: 6482.3 qps
[07/22/2024-02:02:08] [I] Latency: min = 0.155762 ms, max = 0.977173 ms, mean = 0.159906 ms, median = 0.158081 ms, percentile(90%) = 0.159668 ms, percentile(95%) = 0.161133 ms, percentile(99%) = 0.17627 ms
[07/22/2024-02:02:08] [I] Enqueue Time: min = 0.0668945 ms, max = 0.166992 ms, mean = 0.0760865 ms, median = 0.0698242 ms, percentile(90%) = 0.107178 ms, percentile(95%) = 0.109741 ms, percentile(99%) = 0.115479 ms
[07/22/2024-02:02:08] [I] H2D Latency: min = 0.00366211 ms, max = 0.0285034 ms, mean = 0.00416142 ms, median = 0.00402832 ms, percentile(90%) = 0.00439453 ms, percentile(95%) = 0.00463867 ms, percentile(99%) = 0.00585938 ms
[07/22/2024-02:02:08] [I] GPU Compute Time: min = 0.149414 ms, max = 0.969727 ms, mean = 0.152529 ms, median = 0.150574 ms, percentile(90%) = 0.151611 ms, percentile(95%) = 0.152588 ms, percentile(99%) = 0.166992 ms
[07/22/2024-02:02:08] [I] D2H Latency: min = 0.00244141 ms, max = 0.0283203 ms, mean = 0.00321606 ms, median = 0.00292969 ms, percentile(90%) = 0.00408936 ms, percentile(95%) = 0.00418091 ms, percentile(99%) = 0.00463867 ms
[07/22/2024-02:02:08] [I] Total Host Walltime: 3.00048 s
[07/22/2024-02:02:08] [I] Total GPU Compute Time: 2.96669 s
```

Even if the input images to the ResNet model are small (32 x 32) and the batch size is only 1, comparing to the floating-point ResNet engine, the INT8-quantized ResNet engine has a 1.3x latency improvement. The models that have higher math utilization will have more significant latency improvements when quantized to INT8.

### Validate TensorRT Engine

The correctness of the INT8-quantized ResNet TensorRT engine will be validated using a custom TensorRT inference Python script. If the TensorRT engine was built correctly, the accuracy of the INT8-quantized ResNet engine should match the PyTorch INT8-quantized model accuracy.

```bash
$ python resnet_tensorrt.py
Input Tensor:
Tensor Name: x.1 Shape: (1, 3, 32, 32) Data Type: float32 Data Format: TensorFormat.LINEAR
Output Tensor:
Tensor Name: 755 Shape: (1, 10) Data Type: float32 Data Format: TensorFormat.LINEAR
Evaluation Accuracy: 0.8532
```

The accuracy of the INT8-quantized ResNet TensorRT engine matches the PyTorch INT8-quantized model accuracy, suggesting that the INT8-quantized TensorRT engine was built correctly.

## References

- [PyTorch Static Quantization](https://leimao.github.io/blog/PyTorch-Static-Quantization/)
- [PyTorch Eager Mode Quantization TensorRT Acceleration](https://leimao.github.io/blog/PyTorch-Eager-Mode-Quantization-TensorRT-Acceleration/)
