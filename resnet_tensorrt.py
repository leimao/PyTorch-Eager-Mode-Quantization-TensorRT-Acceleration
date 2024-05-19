import numpy as np

import common
import common_runtime

import torch
import torchvision
from torchvision import transforms


def prepare_dataloader(num_workers=8,
                       train_batch_size=128,
                       eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="data",
                                             train=True,
                                             download=True,
                                             transform=train_transform)
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(root="data",
                                            train=False,
                                            download=True,
                                            transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=eval_batch_size,
                                              sampler=test_sampler,
                                              num_workers=num_workers)

    return train_loader, test_loader


def main():

    engine_file_path = "saved_models/resnet_cifar10_int8.engine"

    engine = common_runtime.load_engine(engine_file_path)

    _, test_loader = prepare_dataloader(num_workers=8,
                                        train_batch_size=128,
                                        eval_batch_size=1)

    # Profile index is only useful when the engine has dynamic shapes.
    inputs, outputs, bindings, stream = common.allocate_buffers(
        engine=engine, profile_idx=None)

    # Print input tensor information.
    print("Input Tensor:")
    for host_device_buffer in inputs:
        print(
            f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
            f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
        )
    # Print output tensor information.
    print("Output Tensor:")
    for host_device_buffer in outputs:
        print(
            f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
            f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
        )

    # Execute the engine.
    context = engine.create_execution_context()

    running_corrects = 0
    for test_data, test_label in test_loader:
        # Tensor to Numpy array.
        test_data = test_data.numpy().flatten()
        test_label = test_label.numpy().flatten()
        np.copyto(inputs[0].host, test_data)
        common.do_inference_v2(context,
                               bindings=bindings,
                               inputs=inputs,
                               outputs=outputs,
                               stream=stream)
        # Convert output to Numpy array.
        output_label = np.frombuffer(outputs[0].host,
                                     dtype=np.float32).reshape(-1, 10)
        preds = np.argmax(output_label, axis=1)
        running_corrects += np.sum(preds == test_label)

    eval_accuracy = running_corrects / len(test_loader.dataset)

    # Clean up.
    common.free_buffers(inputs=inputs, outputs=outputs, stream=stream)

    print(f"Evaluation Accuracy: {eval_accuracy}")


if __name__ == "__main__":

    main()
