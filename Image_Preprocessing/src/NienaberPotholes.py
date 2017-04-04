import cv2
import numpy as np  
from matplotlib import pyplot as plt

img = cv2.imread('/home/akanksha/Desktop/MP/Class Project/Dataset/road_cam_34.jpg')

### RESIZING IMAGE
img = cv2.resize(img, (480, 360))

### CONVERTING INTO HSV CHANNEL AND USING THE SATURATION CHANNEL
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

### GAUSSIAN FILTER
gaussian = cv2.GaussianBlur(s, (3,3), 0, 0)

### CANNY FILTER
canny = cv2.Canny(gaussian, 50, 200)

### DILATION 
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(canny, kernel, iterations = 4)

### CONTOUR DETECTION
temp = dilation.copy()
im2, contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (255,0,255), 3)


fig = plt.figure
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(gaussian)
plt.subplot(223)
plt.imshow(dilation)
plt.subplot(224)
plt.imshow(canny)

plt.show()

