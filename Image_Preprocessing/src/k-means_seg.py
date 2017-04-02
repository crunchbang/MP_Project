import numpy as np 
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

def apply_kmeans(img):
	gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	im_1 = gray.reshape(-1)
### Converting the datatype of the column vector to float32
	im_1 = np.float32(im_1)

### Defining the termination criteria for K-Means where TERM_CRITERIA_EPS defines accuracy
### and TERM_CRITERIA_MAX_ITER determines the number of iterations to be run before termination
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	
### Number of Clusters
	K = 2
### Returns the compactness of the data, labels of data(to corresponding clusters) and centroids of the respective clusters
	ret,label,center = cv2.kmeans(im_1,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

### Reforming the image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((gray.shape))

### Applying Gaussian Blur and Canny edge detection for ground truth comparisom
	res3 = cv2.GaussianBlur(res2,(3,3),0)
	edges = cv2.Canny(res3, 100, 200)
	#cv2.imwrite('Result_for_kmeans6.jpg',edges)

### Plotting the results
	plt.subplot(121),plt.imshow(gray,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()

#cv2.waitKey(0)
#cv2.destroyAllWindows()





mypath='/home/tarini/Documents/Pothole-Detection/Data'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
	 images[n] = cv2.imread( join(mypath,onlyfiles[n]) )


for n in range(0,len(onlyfiles)):
	im = apply_kmeans(images[n])
