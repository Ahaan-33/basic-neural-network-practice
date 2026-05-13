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
#class_names = dataset.classes
#print(class_names)
#print(dataset.class_to_idx)
#image, label = dataset[150] # Get a sample image and its label
#print(label)
#print(image.shape)

#plt.imshow(image.squeeze(), cmap='gray')
#lt.title(f'Label: {class_names[label]}')
#plt.show()

#separating and shuffling training and testing datasets using pytorch's random_split function
from torch.utils.data import random_split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Creating training loop
from model import ShapeClassifier

model = ShapeClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 10
loss_values = []
model.train() # Set the model to training mode
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    loss_values.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# create loss curve over epochs after training
epochs = range(1, num_epochs + 1)
plt.figure()
plt.plot(epochs, loss_values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)



#evaluate the model on the test dataset
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')


#model done! lets save it
torch.save(model.state_dict(), 'shape_classifier.pth')
print("Model saved successfully as shape_classifier.pth")

