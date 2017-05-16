
import numpy as np
import argparse
import imutils
import cv2
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256])
 
    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
 
    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)
 
    # return the flattened histogram as the feature vector
    return hist.flatten()

def booya(img):
    resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # resized = img
    width, length, c = resized.shape
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    test = np.array(resized, copy=True)
    roi_top_left_x = int(length / 2 + length / 12)
    roi_top_left_y = int(width - width / 8)
    roi_top_left = (roi_top_left_x, roi_top_left_y)

    roi_bottom_right_x = int(length / 2 + 2 * length / 12)
    roi_bottom_right_y = width - 50
    roi_bottom_right = (roi_bottom_right_x, roi_bottom_right_y)

    cv2.rectangle(test, roi_top_left, roi_bottom_right, (0, 255, 0), 15)
    # print(roi_top_left - roi_bottom_right)

    # roi = resized[1600:1700, 2000:2100]
    roi = resized[roi_top_left_y:roi_bottom_right_y, roi_top_left_x:roi_bottom_right_x]
    #print(roi.shape)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(roi_hsv)
    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    std_h = np.std(h)
    std_s = np.std(s)
    std_v = np.std(v)

    lower_threshold = np.array([mean_h - 3 * std_h, mean_s - 3 * std_s, mean_v - 3 * std_v])
    upper_threshold = np.array([mean_h + 3 * std_h, mean_s + 3 * std_s, mean_v + 3 * std_v])
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    mask_cpy = np.array(mask, copy=True)
    # closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (15, 15))
    dilate = cv2.dilate(mask, (15, 15), iterations=1)
    thresh = cv2.bitwise_and(resized, resized, mask=mask)

    im2, contour, hierarchy = cv2.findContours(mask_cpy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # points = []
    # for cnt in contour:
    #     hull = cv2.convexHull(contour[0])
    #     points.extend(hull)
    # hull = cv2.convexHull(np.array(points))
    cnt = max(contour, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)

    new_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(new_mask, hull, 1)
    result = cv2.bitwise_and(resized, resized, mask=new_mask)
    #fig = plt.figure()

    images = [test, roi, resized, mask, result]
    # titles = [""]

    return result
    #for i in range(len(images)):
    #   ax = fig.add_subplot(3, 2, i + 1)
    #.  ax.set_title(titles[i])
    #    ax.imshow(images[i], cmap="gray")
    #    plt.axis("off")

    #plt.show()


# root = "Train Data/Positive Data/"
rawImages = []
features = []
labels = []

positive = "FinalBooysenDataset/Positive Data/"
negative = "FinalBooysenDataset/Negative Data/"
#train_neg = "Subset 1 (Simplex)/Train Data/Nagative Data"
#test = "Train Data/Positive Data/G0010123.JPEG"

#images = os.listdir(root)
paths = [positive,negative]
for p in paths:
    images = os.listdir(p)
    for i in images:

        img = cv2.imread(p + i)
        img = img[:1800]
        img_new = booya(img)
        
        pixel = image_to_feature_vector(img_new)
        feature = extract_color_histogram(img_new)
        label = i.split('_')[0]


        labels.append(label)
        rawImages.append(pixel)
        features.append(feature)









#for p in train_labels_path:
#        lines = [line.rstrip('\n') for line in open(p)]
 #       for line in lines:
  #          l = line.split(' ')
   #         trainL.append(l[1])

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

test_image = "Subset 1 (Simplex)/Test Data/G0011476.JPEG "
new_test_image = booya(test_image[:1800])
test_feature = (extract_color_histogram(new_test_image))
features.append(test_feature)

print(rawImages.shape)
print(labels.shape)

(trainRI, testRI, trainRL, testRL) = train_test_split(
    rawImages, labels, test_size=0.25, random_state=42)

(trainF, testF, trainFL, testFL) = train_test_split(
    features, labels, test_size=0.25, random_state=42)

#Logistic Regression
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(trainF,trainFL)
acc = logreg.score(testF,testFL)
print("[LR Accuracy: {:.2f}%".format(acc * 100))
#logreg.predict(testImage)

#SVM
model = svm.SVC()
model.fit(trainF,trainFL)
acc = model.score(testF,testFL)
print("SVM Accuracy: {:.2f}%".format(acc * 100))
model.predict(features[:-1])

#kNN
model = KNeighborsClassifier(n_neighbors=3,
    n_jobs=3)
model.fit(trainF, trainFL)
acc = model.score(testF, testFL)
print("kNN accuracy: {:.2f}%".format(acc * 100))
#model.predict(testImage)

#RandomForest
model = RandomForestClassifier()
model.fit(trainF,trainFL)
acc = model.score(testF,testFL)
#print("Random Forest accuracy: {:.2f}%".format(acc * 100))
#model.predict(testImage)


