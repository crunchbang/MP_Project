import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import linear_model
from imutils import paths
from PIL import Image

### EXTRACTING FEATURES

############################################################################################

def extract_ROI(img, size):
	### Extract the road as region of interest from the image by resizing it.
	### For example: size = (480, 360)
	cap = cv2.resize(img, size)
	grayscale_img = cv2.cvtColor(cap, cv2.COLOR_RGB2GRAY)	
	ignore_mask_color = 255
	height, length, _ = cap.shape
	bottom_left_corner = (0+40	, height)                                     
	bottom_right_corner = (length, height)
	top_left_corner = (0+150, int(height/2) + 20)
	top_right_corner = (length-270, int(height/2) + 20)
	region = [np.array([bottom_left_corner,bottom_right_corner, top_right_corner, top_left_corner], dtype=np.int32)] 	

	mask = np.zeros_like(grayscale_img)                           
	keep_region_color = 255
	cv2.fillPoly(mask, region, ignore_mask_color)                  
	region_selected_image = cv2.bitwise_and(grayscale_img, mask)   
	return region_selected_image


#############################################################################################

def extract_features(img):
	### Extracting the raw features from the image

	### CONVERTING INTO HSV CHANNEL AND USING THE SATURATION CHANNEL
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)

	### GAUSSIAN FILTER
	gaussian = cv2.GaussianBlur(s, (7,7), 0, 0)

	### CANNY FILTER
	canny = cv2.Canny(gaussian, 50, 200)

	### DILATION 
	kernel = np.ones((5,5),np.uint8)
	dilation = cv2.dilate(canny, kernel, iterations = 4)

	### CONTOUR DETECTION
	feature = img.copy()
	im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(feature, contours, -1, (255,0,255), 3)

	hist_features = extract_color_histogram(feature)
	return hist_features

def extract_color_histogram(image):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	bins=(8, 8, 8)

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

	### Normalizing the Histogram
	cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()


###################################################################################################
### Reading the images from a folder


i=1
size = (480, 360)

# grab the list of images that we'll be describing
print("[INFO] Extracting features from images...")
imagePaths = list(paths.list_images('/home/akanksha/Desktop/MP/ClassProject/Codes/MVI_8009'))
#print(imagePaths)
	 
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
#rawImages = []
features = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	cv2.imshow(" ", image)
	label = imagePath.split(os.path.sep)[-1][0]      
	 
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	resized_image_array = extract_ROI(image, size)
	resized_image = cv2.cvtColor(np.array(Image.fromarray(resized_image_array)), cv2.COLOR_GRAY2RGB)
	#print(resized_image.shape)
	image_feature = extract_features(resized_image)
	 
	# update the raw images, features, and labels matricies,
	# respectively
	
	features.append(image_feature)
	labels.append(label)
	#print(labels)
	 
	# show an update every 5 images
	if i > 0 and i % 500 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
	print(i)

### Converting the features list into numpy array and saving them
features_array = np.array(features)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																														
print(features_array.shape)
labels_array = np.array(labels)

### Saving the histogram features in a file
np.savez('potholes_features', features_array, labels_array)
