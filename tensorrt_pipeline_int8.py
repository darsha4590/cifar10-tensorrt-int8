# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 17:08:54 2025

@author: 15138
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:23:03 2025

@author: 15138
"""

import tensorrt as trt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from calibrators import CIFAR10EntropyCalibrator



TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

onnx_path = "simplecnn_cifar10.onnx"

# Create builder, network, parser
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)

parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("Failed to parse ONNX model")


# Create the builder config
config = builder.create_builder_config()

# Set the max workspace size (required)
# This is the memory (in bytes) available for TensorRT to use for optimization.
# 1 << 30 means 1GB.
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# Enable FP16 mode (optional, requires hardware support)
if builder.platform_has_fast_fp16 and False:
    config.set_flag(trt.BuilderFlag.FP16)

# Define optimization profile for dynamic batch size
profile = builder.create_optimization_profile()

# Assume input name is 'input' (you already defined this in ONNX export)
input_name = network.get_input(0).name

# Set min, opt, and max shapes for batch dimension
# This defines the range of input shapes TRT should support for this engine
profile.set_shape(input_name,
                  min=(1, 3, 32, 32),   # smallest batch size
                  opt=(8, 3, 32, 32),   # typical/optimal batch size
                  max=(64, 3, 32, 32))  # largest batch size to support

# Add the profile to the config
config.add_optimization_profile(profile)

# Load the test data
# --------------------
# Transforms
# --------------------
tf_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])


# --------------------
# Dataset & Dataloaders
# --------------------

test_dataset = datasets.DatasetFolder(
    root="cifar10/test",
    loader=datasets.folder.default_loader,
    extensions=['.jpg', '.png', '.jpeg'],
    transform=tf_test
)
test_loader = DataLoader(test_dataset,
                         batch_size=32,
                         shuffle=False,
                         num_workers=0)

# Add int8 calibrator for quantization
calibrator = CIFAR10EntropyCalibrator(test_loader, 32)
config.int8_calibrator = calibrator

# Build the serialized engine (can take time depending on model complexity)
serialized_engine = builder.build_serialized_network(network, config)

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)


# Create execution context for inference
context = engine.create_execution_context()


context.set_input_shape("input", (32, 3, 32, 32))




images, labels = next(iter(test_loader))
# Set shape before anything else
context.set_input_shape("input", (32, 3, 32, 32))

# Define a custom profiler
class MyProfiler(trt.IProfiler):
    def report_layer_time(self, layer_name, ms):
        print(f"{layer_name}: {ms:.3f} ms")
context.profiler = MyProfiler()

# Query expected shapes
in_shape = context.get_tensor_shape("input")
out_shape = context.get_tensor_shape("logits")
print("Engine expects input:", in_shape, "output:", out_shape)

# Allocate
input_trt = np.ascontiguousarray(images.numpy().astype(np.float32))
d_input = cuda.mem_alloc(input_trt.nbytes)

output = np.empty(out_shape, dtype=np.float32)
d_output = cuda.mem_alloc(output.nbytes)

# Transfer
cuda.memcpy_htod(d_input, input_trt)

# Bind addresses
context.set_tensor_address("input", int(d_input))
context.set_tensor_address("logits", int(d_output))

# Run
stream = cuda.Stream()
context.execute_async_v3(stream_handle=stream.handle)
stream.synchronize()

# Copy back
cuda.memcpy_dtoh(output, d_output)
print("Output shape:", output.shape)


