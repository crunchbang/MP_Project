import cv2
from matplotlib import pyplot as plt

orig = cv2.imread('sample.jpg', cv2.IMREAD_COLOR)
blue_channel, green_channel, red_channel = cv2.split(orig)
# OpenCV reads images as (B,G,R) while matplotlib renders images 
# assuming it is in (R,G,B). This transforms the image from (B,G,R) to (R,G,B)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

fig = plt.figure()  # create a plot on to which subplots can be added

channel_image = [orig, red_channel, green_channel, blue_channel]
channel_title = ["Original", "Red channel", "Green channel", "Blue channel"]
# Uncomment the below code snippet to display the image
# using the individual channel colors
# channel_cmap = ["gray", "Reds", "Greens", "Blues"]

# Create a 2x2 plot which will be populated by the loop
# using corresponding image, title and color map

# add each image as subplot of main plot
for i in range(len(channel_image)):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.set_title(channel_title[i])
    # Display the image, use gray scale color map to 
    # display the single channel image with color channel
    # intensity varying from black to white
    ax.imshow(channel_image[i], cmap="gray")
    # Uncomment the below code snippet to display the image
    # using the individual channel colors
    # ax.imshow(channel_image[i], cmap=channel_cmap[i])
    plt.xticks([])  # remove x and y axis markings
    plt.yticks([])

plt.show()
