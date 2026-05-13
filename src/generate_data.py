import os
import numpy as np
import matplotlib.pyplot as plt


#creating a blank 16x16 image
img = np.zeros((16, 16), dtype=np.uint8)

#filling a square in the center of the image
img[4:12, 4:12] = 1
plt.imshow(img, cmap='gray')
plt.show()

#making a function to create a dataset of images with squares in different positions

def generate_square_dataset(num_samples):
    square_dataset = []
    for _ in range(num_samples):
        img = np.zeros((16, 16), dtype=np.uint8)
        x = np.random.randint(0, 8)  # Random x position for the square
        y = np.random.randint(0, 8)  # Random y position for the square
        img[y:y+8, x:x+8] = 1  # Draw a 8x8 square
        square_dataset.append(img)
    return np.array(square_dataset)

# Generate datasets with 100 samples for saving
square_dataset = generate_square_dataset(100)
# Show first 5 samples for visualization
for i in range(5):
    plt.imshow(square_dataset[i], cmap='gray')
    plt.title(f'Sample {i+1}')
    plt.show()

# making a function to create a dataset of images with circles in different positions
def generate_circle_dataset(num_samples):
    circle_dataset = []
    for _ in range(num_samples):
        img = np.zeros((16, 16), dtype=np.uint8)
        x = np.random.randint(4, 12)  # Random x position for the circle center
        y = np.random.randint(4, 12)  # Random y position for the   circle center
        for i in range(16):
            for j in range(16):
                if (i - y) ** 2 + (j - x) ** 2 <= 16:  # Circle with radius 4
                    img[i, j] = 1
        circle_dataset.append(img)
    return np.array(circle_dataset)

# Generate circle dataset with 100 samples for saving
circle_dataset = generate_circle_dataset(100)
# Show first 5 samples for visualization
for i in range(5):
    plt.imshow(circle_dataset[i], cmap='gray')
    plt.title(f'Sample {i+1}')
    plt.show()

# Generate 100 samples for saving
circle_dataset = generate_circle_dataset(100)

# save the generated datasets as images in data directory in respective folders
os.makedirs('data/circle', exist_ok=True)
os.makedirs('data/square', exist_ok=True)
for i, img in enumerate(circle_dataset):
    filename = f'data/circle/circle_{i}.png'
    plt.imsave(filename, img, cmap='gray')
for i, img in enumerate(square_dataset):
    filename = f'data/square/square_{i}.png'
    plt.imsave(filename, img, cmap='gray')





