# CIFAR10 Model Deployment with TensorRT  

This repository demonstrates an **end-to-end deployment pipeline** for deep learning models:  

1. **Model training in PyTorch**  
2. **Exporting to ONNX**  
3. **Inference with ONNX Runtime (ORT)**  
4. **TensorRT FP32 / FP16 / INT8 quantized inference with calibration**  
5. **Profiling inference latency and per-layer performance**  

The target dataset is **CIFAR-10**, and this repo includes training scripts, ONNX export, TensorRT pipelines, and a custom INT8 calibrator.  

---

## 📂 Project Structure  
├── calibrators.py # INT8 entropy calibrator for CIFAR-10
├── dataset.py # Dataset utilities
├── model.py # CNN model definition
├── model_training.py # PyTorch training loop
├── export_to_onnx.py # Export trained PyTorch model → ONNX
├── runme.py # Run ONNX Runtime inference
├── tensorrt_pipeline_fp32.py # TensorRT FP32/FP16 inference
├── tensorrt_pipeline_int8.py # TensorRT INT8 inference with calibration

1. Train the Model
python model_training.py


Trains a simple CNN on CIFAR-10

Saves weights → simplecnn_cifar10.pth

2. Export to ONNX
python export_to_onnx.py


Converts PyTorch model → simplecnn_cifar10.onnx

3. Inference with ONNX Runtime
python runme.py


Runs inference with ONNX Runtime

Prints predictions and average latency

4. TensorRT Inference
FP32 / FP16
python tensorrt_pipeline_fp32.py


Builds TensorRT engine in FP32 (or FP16 if hardware supports)

Runs inference on CIFAR-10

INT8 with Calibration
python tensorrt_pipeline_int8.py


Uses CIFAR10EntropyCalibrator to calibrate INT8 scales

Builds INT8 engine & runs inference

Prints layer-wise profiling
