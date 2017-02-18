import cv2
import numpy as np
from matplotlib import pyplot as plt

# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sinusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold

# create the gabor kernel
g_kernel = cv2.getGaborKernel((31, 31), 3.0, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F)

img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
fig = plt.figure()

images = [img, g_kernel, filtered_img]
titles = ["Original", "Gabor kernel", "filtered image"]

for i in range(len(images)):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.set_title(titles[i])
    ax.imshow(images[i], cmap="gray")
    plt.axis("off")

plt.show()
