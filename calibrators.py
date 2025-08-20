# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:59:09 2025

@author: 15138
"""

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class CIFAR10EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, batch_size, cache_file = "cifar_calib.cache"):
        super().__init__()
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.data_iter = iter(self.dataloader)
        self.batch_size = batch_size
        self.device_input = cuda.mem_alloc(trt.volume((batch_size, 3, 32, 32)) * np.float32().nbytes)
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        try:
            images, _ = next(self.data_iter)   # only inputs needed
            images_np = images.numpy().astype(np.float32).ravel()
            cuda.memcpy_htod(self.device_input, images_np)
            return [int(self.device_input)]
        except StopIteration:
            return None
    def read_calibration_cache(self):
        # If calibration is done use it
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
    
        
