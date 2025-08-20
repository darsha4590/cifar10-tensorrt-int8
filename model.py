# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:55:03 2025

@author: 15138
"""

import torch
import torch.nn as nn

class SimpleCNNForCifar10(nn.Module):
    def __init__(self, num_classes = 10, input_shape=(3, 32, 32)):
        super(SimpleCNNForCifar10, self).__init__()
        
        self.Sequential_Compute = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16, eps = 1e-5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64, eps = 1e-5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128, eps = 1e-5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32, eps = 1e-5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        
        # Automatically compute flatten size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.Sequential_Compute(dummy_input)
            flatten_size = dummy_output.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(flatten_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.Sequential_Compute(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x
    
        