import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/akanksha/Desktop/MP/Class Project/Dataset/DSC_2524.JPG')
img = cv2.resize(img, (480, 360))


### SEGMENTATAION USING MEANSHIFT

### DEFINING PARAMETERS 

### Spatial Window Radius
sp = 25
### Color Window Radius
sr = 45
### Maximum Level of pyramid for segmentation
maxLevel = 10

res = cv2.pyrMeanShiftFiltering(img, sp, sr, maxLevel)
gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_res, (3, 3), 0)
output = cv2.Canny(blur, 100, 200)

temp = img.copy()

### CONTOUR DETECTION
im2, contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(temp, contours, -1, (255,0,255), 3)

#cv2.imshow('Output', temp)
#cv2.imwrite('Meanshift_Contours.jpg', temp)

fig = plt.figure
plt.subplot(2,2,1),plt.imshow(img)
plt.title('Original Image')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,2),plt.imshow(res)
plt.title('Meanshift Applied')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,3),plt.imshow(output)
plt.title('Canny Edge Detected')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,4),plt.imshow(temp)
plt.title('Cantours Detected')
plt.xticks([]),plt.yticks([])


plt.show()

cv2.waitKey(0)

