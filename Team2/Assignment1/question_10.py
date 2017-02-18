import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
# Laplacian of Gaussian
img = cv2.GaussianBlur(img, (3, 3), 0)
lap = cv2.Laplacian(img, -1)
img = img.astype(np.int16)
# sharpen the original image
shp1 = img - lap
# make all negative values 0
shp1 = shp1.clip(min=0)

channel_image = [img, lap, shp1, ]
channel_title = ["Original", "Laplacian", "Sharpened", ]

fig = plt.figure()

for i in range(len(channel_image)):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.set_title(channel_title[i])
    ax.imshow(channel_image[i], cmap="gray")
    plt.axis("off")

plt.show()
