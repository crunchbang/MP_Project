import cv2
import numpy as np
from matplotlib import pyplot as plt

# function to perform lane detection
def detectRoads(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # smoothing to remove noise
    blur = cv2.GaussianBlur(img_gr, (3, 3), 0)
    # detect edges with sobel-x
    sbl = cv2.Sobel(blur, -1, 1, 0, ksize=3)
    # threshold to remove low intensity edges
    sbl[sbl < 100] = 0
    # detect lines with probabilistic HLT
    lines = cv2.HoughLinesP(sbl, 1, np.pi / 180, 200, minLineLength=10, maxLineGap=5)

    # plotting the lanes on original image
    for line in lines:
        [[x1, y1, x2, y2]] = line
        cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return  img_rgb


roads = ["road1.png", "road2.png"]
channel_title = ["Original", "Result"]

fig = plt.figure()
k = 1
# display the results
for road in roads:
    img = cv2.imread(road, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    channel_image = detectRoads(img)
    ax = fig.add_subplot(2, 2, k)
    ax.imshow(img_rgb, cmap="gray")
    ax.set_title(channel_title[0])
    plt.axis("off")
    k = k + 1
    ax = fig.add_subplot(2, 2, k)
    ax.set_title(channel_title[1])
    ax.imshow(channel_image, cmap="gray")
    k = k + 1
    plt.axis("off")

plt.show()
