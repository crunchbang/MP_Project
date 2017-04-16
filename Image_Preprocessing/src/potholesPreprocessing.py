import cv2
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('/home/akanksha/Desktop/MP/Class Project/Dataset/road_1.jpg', 0)
img = cv2.resize(img, (480, 360))
### BLURRING 

gaussian_blur = cv2.GaussianBlur(img, (3, 3), 0)
#median_blur = cv2.medianBlur(img, 3)

### THRESHOLDING
ret, thresh_g = cv2.threshold(gaussian_blur, 127, 255, 0)
#ret, thresh_m = cv2.threshold(median_blur, 127, 255, 0)

### IMAGE GRADIENTS
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

temp = img.copy()


### CONTOUR DETECTION
im2, contours, hierarchy = cv2.findContours(thresh_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(temp, contours, -1, (255,0,255), 3)


### MORPHOLOGICAL TRANSFORMATION
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
#dilation = cv2.dilate(img,kernel,iterations = 1)

fig = plt.figure
plt.subplot(2,2,1),plt.imshow(img)
plt.title('Original Image')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,2),plt.imshow(erosion)
plt.title('Erosion')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,3),plt.imshow(sobely)
plt.title('Sobel Y')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,4),plt.imshow(temp)
plt.title('Contours Detected')
plt.xticks([]),plt.yticks([])

plt.show()

cv2.waitKey(0)