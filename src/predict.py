#load saved model
import torch
from model import ShapeClassifier  

#load the saved model 
model = ShapeClassifier()
model.load_state_dict(torch.load('shape_classifier.pth'))
model.eval()  # Set the model to evaluation mode
# Now you can use the model for inference
# Example: Predicting the class of a new image
#import a single image from the test dataset and predict its class
from torchvision import datasets, transforms
# Define the transformation to convert images to tensors
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor()  # Convert to tensor
])  
# Load the test dataset
test_dataset = datasets.ImageFolder(root='data/', transform=transform)
# Get a sample image and its label from the test dataset
image, label = test_dataset[120]  # Change the index to test different images
# Add a batch dimension to the image tensor
image = image.unsqueeze(0)  # Shape: [1, 1, 16, 16]
# Predict the class of the image
with torch.no_grad():  # Disable gradient calculation for inference
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()
print(f'Predicted class: {predicted_class}, Actual class: {label}')
