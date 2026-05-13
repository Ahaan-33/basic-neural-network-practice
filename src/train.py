from torchvision import datasets, transforms
import torch
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
#plt.show()

#separating and shuffling training and testing datasets
from sklearn.model_selection import train_test_split
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Creating training loop
from model import ShapeClassifier
model = ShapeClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 10
loss_values = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

#analyse the results by plotting the loss curve
loss_values.append(running_loss/len(dataloader))

plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

