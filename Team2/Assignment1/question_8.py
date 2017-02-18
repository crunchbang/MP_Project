import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)

# Prewitt kernels
prewitt_kernelx = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
prewitt_kernely = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])

# Apply the kernel
prewittx = cv2.filter2D(img, cv2.CV_64F, prewitt_kernelx)
prewitty = cv2.filter2D(img, cv2.CV_64F, prewitt_kernely)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

images = [img, sobelx, sobely, prewittx, prewitty]
titles = ["Original", "Sobel X", "Sobel Y", "Prewitt X", "Prewitt Y"]

fig = plt.figure()

for i in range(len(images)):
    ax = fig.add_subplot(3, 2, i + 1)
    ax.set_title(titles[i])
    ax.imshow(images[i], cmap="gray")
    plt.axis("off")

plt.show()
