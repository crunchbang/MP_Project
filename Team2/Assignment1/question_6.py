import cv2
from matplotlib import pyplot as plt

orig = cv2.imread("noise3.jpg", cv2.IMREAD_GRAYSCALE)
#orig = cv2.imread("noise2.jpg", cv2.IMREAD_GRAYSCALE)

# Gaussian blur for different kernel size
gaussian_blur_3 = cv2.GaussianBlur(orig, (3, 3), 0)
gaussian_blur_5 = cv2.GaussianBlur(orig, (5, 5), 0)
gaussian_blur_7 = cv2.GaussianBlur(orig, (7, 7), 0)

fig = plt.figure()

images = [orig, gaussian_blur_3, gaussian_blur_5, gaussian_blur_7]
titles = ["Original", "3", "5", "7"]

for i in range(len(images)):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.set_title(titles[i])
    ax.imshow(images[i], cmap="gray")
    plt.axis("off")

plt.show()
