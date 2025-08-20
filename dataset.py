# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:47:13 2025

@author: 15138
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FashionMNISTDataset(Dataset):
    def __init__(self,  file_path = None, transform = None):
        if not file_path:
            print("Data path needs to provided")
            
        self.dataframe = pd.read_csv(file_path)
        print(self.dataframe)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx, 0]
        image = self.dataframe.iloc[idx, 1:].values.astype(np.uint8).reshape(28,28)
        if self.transform:
            image = self.transform(image)
        
        image = torch.tensor(image, dtype = torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype = torch.long)
        
        return image, label
            
    
            
        