# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:27:55 2025

@author: 15138
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import onnx
from onnx import checker, shape_inference
import onnxruntime as ort
from model import SimpleCNNForCifar10
import numpy as np
import time


# Lets load the model 

model = SimpleCNNForCifar10()

model.load_state_dict(torch.load("simplecnn_cifar10.pth", map_location = "cpu", weights_only= True))
model.eval()
model.cpu()

# Creating dummy data for onnx object creation
dummy_input = torch.rand(1,3,32,32, dtype = torch.float32)

onnx_path = "simplecnn_cifar10.onnx"

input_names = ["input"]
output_names = ["logits"]
# Defining this to tell onxx which axes shape could be modified
dynamic_axes = {
    "input": {0: "batch"},
    "logits": {0: "batch"}
}

# Use torch.onnx.export and save the .onnx file for inference through onnxruntime
with torch.inference_mode():
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

# Lets check the saved model 
m = onnx.load(onnx_path)
checker.check_model(m)

# Check if the pytorch model and onnx model get the same inference results

onnx_session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inp_name = onnx_session.get_inputs()[0].name
out_name = onnx_session.get_outputs()[0].name

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
                         batch_size=256,
                         shuffle=False,
                         num_workers=0)

images, labels = next(iter(test_loader))
images_torch = images.clone()
images_onnx = images.numpy()

with torch.no_grad():
    model.eval()
    logits_torch = model(images_torch).detach().numpy()

logits_onnx = onnx_session.run([out_name], {inp_name: images_onnx})[0]


# Compare top-1
top1_pt  = logits_torch.argmax(axis=1)
top1_ort = logits_onnx.argmax(axis=1)
agree = (top1_pt == top1_ort).mean()
print(f"Top-1 agreement: {agree*100:.2f}%")


mse_logits = np.mean((logits_onnx - logits_torch) ** 2)
print(f"MSE (logits): {mse_logits:.6e}")

# Get a batch
images, _ = next(iter(test_loader))
images_onnx = images.numpy()

# Warmup (GPU lazy init)
for _ in range(10):
    _ = onnx_session.run([out_name], {inp_name: images_onnx})

# Measure latency (ms per batch)
num_runs = 50
start = time.time()
for _ in range(num_runs):
    _ = onnx_session.run([out_name], {inp_name: images_onnx})
end = time.time()

avg_batch_time = (end - start) / num_runs * 1000  # ms
per_image_time = avg_batch_time / images_onnx.shape[0]
throughput = images_onnx.shape[0] / (avg_batch_time / 1000)  # images/sec

print(f"Batch size: {images_onnx.shape[0]}")
print(f"Average batch time: {avg_batch_time:.2f} ms")
print(f"Average per-image latency: {per_image_time:.4f} ms")
print(f"Throughput: {throughput:.2f} images/sec")

























