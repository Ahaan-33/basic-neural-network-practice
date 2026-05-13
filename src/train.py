import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

#transform to convert images to tensors
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor()  # Convert to tensor
])

#loading the datasets
dataset = datasets.ImageFolder(root='data/', transform=transform)

#Dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#check shape of images and labels
for images, labels in dataloader:
    print(images.shape)  # Should be [batch_size, 1, 16, 16]
    print(labels.shape)  # Should be [batch_size]
    break

#to check if both classes are present in the dataset
class_names = dataset.classes
print(class_names)
print(dataset.class_to_idx)
image, label = dataset[150] # Get a sample image and its label
print(label)
print(image.shape)
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {class_names[label]}')
plt.show()



