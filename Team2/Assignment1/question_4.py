import cv2
from matplotlib import pyplot as plt

img_bgr = cv2.imread("sample.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) # Convert to grayscale

fig = plt.figure()

image = [img_rgb, img_gray]
title = ["Original", "Grayscale"]

for i in range(len(image)):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(title[i])
    ax.imshow(image[i], cmap="gray")
    plt.xticks([])
    plt.yticks([])

plt.show()
