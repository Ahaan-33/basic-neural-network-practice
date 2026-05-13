import torch.nn as nn
import torch

#define the structure of the CNN shape classifier model
class ShapeClassifier(nn.Module): 
    def __init__(self):
        super(ShapeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Input: 1 channel, Output: 16 channels
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with a 2x2 window
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Input: 16 channels, Output: 32 channels
        self.fc1 = nn.Linear(32 * 4 * 4, 64)  # Fully connected layer (32 channels * 4x4 feature map)
        self.fc2 = nn.Linear(64, 2)  # Output layer for binary classification (square vs circle)

        #forward function 
    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)  # Debugging: Check shape after first convolution
        x = nn.ReLU()(x)  # Activation function
        x = self.pool(x)
        x = self.conv2(x)
        print(x.shape)  # Debugging: Check shape after second convolution
        x = nn.ReLU()(x)  # Activation function
        x = self.pool(x)
        x = x.view(-1, 32 * 4 * 4)  # Flatten the feature map
        print (x.shape)  # Debugging: Check shape before fully connected layers

        x = self.fc1(x)
        x = nn.ReLU()(x)  # Activation function
        x = self.fc2(x)
        return x
    
#run the model to check using train.py
if __name__ == "__main__":
    model = ShapeClassifier()
    print(model)
    # Create a dummy input tensor with the shape of a single grayscale image (1, 1, 16, 16)
    dummy_input = torch.randn(32, 1, 16, 16) 
    output = model(dummy_input)
    print("image shape:", dummy_input.shape)  # Should be [1, 1, 16, 16]
    print("Output shape:", output.shape)  # Should be [1, 2]
    print("Output:", output)

#Training the model will be done in train.py, this file is just for defining the model structure and testing it with dummy data.

