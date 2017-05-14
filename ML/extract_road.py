import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def booya(img):
    # resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    resized   = img
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    # width, length, c = hsv.shape
    roi = resized[1600:1700, 2000:2100]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(roi_hsv)
    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    std_h = np.std(h)
    std_s = np.std(s)
    std_v = np.std(v)

    lower_threshold = np.array([mean_h - std_h, mean_s - std_s, mean_v - std_v])
    upper_threshold = np.array([mean_h + std_h, mean_s + std_s, mean_v + std_v])
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    thresh = cv2.bitwise_and(resized, resized, mask=mask)

    im2, contour, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contour:
        hull = cv2.convexHull(contour[0])
        points.extend(hull)
    hull = cv2.convexHull(np.array(points))
    cv2.drawContours(resized, hull, -1, (0, 255, 0), 3)
    fig = plt.figure()

    images = [resized, mask, img]
    # titles = [""]

    for i in range(len(images)):
        ax = fig.add_subplot(3, 2, i + 1)
        # ax.set_title(titles[i])
        ax.imshow(images[i], cmap="gray")
        plt.axis("off")

    plt.show()


root = "Train Data/Positive Data/"
test = "Train Data/Positive Data/G0010123.JPG"
images = os.listdir(root)
path = test
for i in images:
    img = cv2.imread(root + i)
    img = img[:1800]
    booya(img)
# booya(img)
