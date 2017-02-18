import cv2
from matplotlib import pyplot as plt

img_bgr = cv2.imread("sample.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
# cv2.imshow renders the Lab image as (B,G,R) instead of (R,G,B)
# Following code changes the order of channels to get the same output
# in matplotlib
img_lab_cv = img_lab[:, :, ::-1]

fig = plt.figure()

image = [img_rgb, img_lab_cv]
title = ["Original", "L*a*b*"]

for i in range(len(image)):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(title[i])
    ax.imshow(image[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
