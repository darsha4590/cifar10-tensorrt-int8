# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:42:07 2025

@author: 15138
"""

from torch.utils.data import DataLoader
from model import SimpleCNNForCifar10
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import StepLR

# Import the datasets for evaluation and training pipelines


tf_test = transforms.Compose([transforms.ToTensor(), 
                              transforms.Normalize(mean= [0.4914,0.4822,0.4465], std = [0.2023,0.1994,0.2010])
                              ])

tf_train = transforms.Compose([transforms.RandomCrop(32, padding = 4),
                               transforms.RandomHorizontalFlip(p = 0.5),
                               transforms.ToTensor(), 
                               transforms.Normalize(mean= [0.4914,0.4822,0.4465], std = [0.2023,0.1994,0.2010])
                               ])


train_dataset = datasets.DatasetFolder(root = "cifar10/train", loader = datasets.folder.default_loader, extensions= ['.jpg', '.png', '.jpeg'], transform= tf_train)
train_loader = DataLoader(train_dataset, batch_size= 64, shuffle= True)
test_dataset = datasets.DatasetFolder(root = "cifar10/test", loader = datasets.folder.default_loader, extensions= ['.jpg', '.png', '.jpeg'], transform= tf_test)
test_loader = DataLoader(test_dataset, batch_size= 256, shuffle= False)


# Creating the model for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNNForCifar10().to(device)


# Model Training
activation = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
scheduler = StepLR(optimizer, step_size= 5, gamma = 0.5)

# Writing the training loop
for epoch in range(30):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        images = images.to(device, non_blocking = True)
        labels= labels.to(device, non_blocking = True)
        outputs = model(images)
        loss = activation(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate on test dataset

correct = 0
total = 0
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking = True)
        labels= labels.to(device, non_blocking = True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100.0 * correct/(total*1.0)

print(f'Test Accuracy: {accuracy:.2f}%')

# --------------------
# Save model for export later
# --------------------
torch.save(model.state_dict(), "simplecnn_cifar10.pth")