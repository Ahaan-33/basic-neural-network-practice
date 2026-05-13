import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((16, 16), dtype=np.uint8)

img[:, :8] = 1
plt.imshow(img, cmap='gray')
plt.show()
