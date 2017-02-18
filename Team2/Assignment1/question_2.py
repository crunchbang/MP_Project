import cv2
from matplotlib import pyplot as plt

# Reading the image
img_bgr = cv2.imread('sample.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB Channel

img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV Channel
hue, saturation, variance = cv2.split(img_hsv)  # Split H,S,V components

channel_image = [img, hue, saturation, variance]
channel_title = ["Original", "Hue channel", "Saturation channel", "Value channel"]
channel_cmap = ["gray", "hsv", "gray", "gray"]

# Display the observations
fig = plt.figure()

for i in range(len(channel_image)):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.set_title(channel_title[i])
    ax.imshow(channel_image[i], cmap=channel_cmap[i])
    plt.xticks([])
    plt.yticks([])

img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)  # Convert to HSL Channel
hue, luminance, saturation = cv2.split(img_hls)  # Split H,L,S Channel

channel_image = [img, hue, saturation, luminance]
channel_title = ["Original", "Hue channel", "Saturation channel", "Luminance channel"]
channel_cmap = ["gray", "hsv", "gray", "gray"]

# Display the observations
fig = plt.figure()

for i in range(len(channel_image)):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.set_title(channel_title[i])
    ax.imshow(channel_image[i], cmap=channel_cmap[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
