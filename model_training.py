# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:42:07 2025
Author: 15138
Description: CIFAR-10 training script for SimpleCNNForCifar10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets
from model import SimpleCNNForCifar10
import random
import numpy as np

# --------------------
# Reproducibility
# --------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------
# Transforms
# --------------------
tf_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

# --------------------
# Dataset & Dataloaders
# --------------------
train_dataset = datasets.DatasetFolder(
    root="cifar10/train",
    loader=datasets.folder.default_loader,
    extensions=['.jpg', '.png', '.jpeg'],
    transform=tf_train
)
train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True,
                          drop_last=True)

test_dataset = datasets.DatasetFolder(
    root="cifar10/test",
    loader=datasets.folder.default_loader,
    extensions=['.jpg', '.png', '.jpeg'],
    transform=tf_test
)
test_loader = DataLoader(test_dataset,
                         batch_size=256,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True)

# --------------------
# Device & Model
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNNForCifar10().to(device)

# --------------------
# Loss, Optimizer, Scheduler
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# --------------------
# Training Loop
# --------------------
for epoch in range(30):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")

# --------------------
# Evaluation
# --------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# --------------------
# Save model for export later
# --------------------
torch.save(model.state_dict(), "simplecnn_cifar10.pth")
