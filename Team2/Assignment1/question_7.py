import cv2
import numpy as np
from matplotlib import pyplot as plt

orig = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
noisy_img = orig.copy()

# add salt and pepper noise
# There are multiple ways to do it, this being one of them
# choose a random value in the range 0 - 0.05, the
# probablity of there being noise in a pixel
p = np.random.uniform(0, 0.05)
# create a noise matrix of the same dimension as the image with
# values uniformly distributed in the range [0, 1)
rand_noise = np.random.rand(*orig.shape)
# add noise (make the pixel black or white) at locations of the original
# image where the conditions are satisfied
noisy_img[rand_noise < p] = 0
noisy_img[rand_noise > 1 - p] = 255

filtered_img = cv2.medianBlur(noisy_img, 3)

fig = plt.figure()

images = [orig, noisy_img, filtered_img]
titles = ["Original", "Salt and Pepper noise", "filtered image"]

for i in range(len(images)):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.set_title(titles[i])
    ax.imshow(images[i], cmap="gray")
    plt.axis("off")

plt.show()
