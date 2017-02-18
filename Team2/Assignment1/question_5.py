import cv2
import numpy as np
from matplotlib import pyplot as plt

orig = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)

hist_equal = cv2.equalizeHist(orig)

# numpy does elementwise scalar operation on the image matrix
whitened_img = (orig - np.mean(orig)) / np.std(orig)

fig = plt.figure()

images = [orig, whitened_img, hist_equal]
titles = ["Original", "Whitened", "Histogram equalization"]
pos = 1

# display image and the corresponding historgram
for i in range(len(images)):
    ax = fig.add_subplot(3, 2, pos)
    ax.set_title(titles[i])
    # uncomment to see the output like in cv2.imshow()
    # if (images[i] == whitened_img).all():
    #     ax.imshow(images[i], cmap="gray", vmax=1, vmin=0)
    # else:
    #     ax.imshow(images[i], cmap="gray")
    ax.imshow(images[i], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    pos += 1
    ax = fig.add_subplot(3, 2, pos)
    # round to 2 decimal places
    mean = round(np.mean(images[i]), 2)
    std = round(np.std(images[i]), 2)
    hist_title = "Mean:" + str(mean) + "  Std:" + str(std)
    ax.set_title(hist_title)
    ax.hist(images[i].ravel(), 256, [0, 256])
    pos += 1
    plt.xticks([])
    plt.yticks([])

plt.show()
