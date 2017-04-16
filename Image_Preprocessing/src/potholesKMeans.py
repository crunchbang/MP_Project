import cv2
import numpy as np      
from matplotlib import pyplot as plt     

img = cv2.imread('/home/akanksha/Desktop/MP/Class Project/Dataset/road_cam_11.jpg', 0)
img = cv2.resize(img, (480, 360))
### Reshaping the image array into a column vector
temp = img.reshape(-1, 3)
### Converting the dataype of pixels as float 32
features = np.float32(temp)

### RUNNING K-MEANS
### Specifying the criteria of intented accuracy and iterations
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)

### Flags to specify how initial centers are taken
flags = cv2.KMEANS_RANDOM_CENTERS

### Number of clusters
K = 3
compactness, labels, centers = cv2.kmeans(features, K, None, criteria, 30, flags)


### Reforming the image
centers = np.uint8(centers)
out = centers[labels.flatten()]
output = out.reshape(img.shape)

### Gaussian Blur
blur = cv2.GaussianBlur(output, (3, 3), 0)

### Canny Edge Detection
seg = cv2.Canny(blur, 50, 50)

### Creating the duplicate copy of the image
temp = img.copy()

### CONTOUR DETECTION
im2, contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(temp, contours, -1, (255,0,255), 3)


fig = plt.figure
plt.subplot(2,2,1),plt.imshow(img)
plt.title('Original Image')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,2),plt.imshow(output)
plt.title('K Means Applied')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,3),plt.imshow(seg)
plt.title('Segmented Image')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,4),plt.imshow(temp)
plt.title('Cantours Detected')
plt.xticks([]),plt.yticks([])

plt.show()





cv2.waitKey(0)