import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# variable for storing traing info
images = []
train_labels = []
train_desc = []

"""
    Data organization into test and train

    data
    ├── classification
    │   ├── test
    │   │   ├── 0030.jpg
    │   │   ├── 0031.jpg
    │   │   ├── 0038.jpg
    │   │   ├── 0039.jpg
    │   │   ├── 0052.jpg
    │   │   ├── 0053.jpg
    │   │   ├── 0054.jpg
    │   │   ├── 0055.jpg
    │   │   ├── horse18.jpg
    │   │   ├── horse19.jpg
    │   │   ├── horse20.jpg
    │   │   ├── horse5 .jpg
    │   │   ├── horse6 .jpg
    │   │   ├── horse7 .jpg
    │   │   └── horse8 .jpg
    │   └── train
    │       ├── Bikes
    │       │   ├── 0001.jpg
    │       │   ├── 0002.jpg
    │       │   ├── 0003.jpg
    │       │   ├── 0004.jpg
    │       │   ├── 0010.jpg
    │       │   ├── 0011.jpg
    │       │   ├── 0078.jpg
    │       │   ├── 0079.jpg
    │       │   └── 0080.jpg
    │       └── Horses
    │           ├── h1.jpg
    │           ├── h2.jpg
    │           ├── horse10.jpg
    │           ├── horse71.jpg
    │           ├── horse79.jpg
    │           ├── horse80.jpg
    │           ├── horse81.jpg
    │           └── horse9 .jpg
"""

# train and test data paths
path_str = "data/classification/train"
path_str_test = "data/classification/test"

path = os.listdir(path_str)
path_test = os.listdir(path_str_test)
images_test = []

label = ["Bike", "Horse"]

# Stage 1.1 : Training
# reading traing data
for paths in range(len(path)):
    if path[paths] == 'Bikes':
        bike_images = os.listdir(path_str+"/"+path[paths])
        for bike_path in range (len(bike_images)):
            str_path =path_str + "/" + path[paths] + "/" + bike_images[
                bike_path]
            images.append(str_path)

            train_labels.append(1)
    if path[paths] == 'Horses':
        horse_images = os.listdir(path_str + "/" + path[paths])
        for horse_path in range(len(horse_images)):
            str_path = path_str + "/" + path[paths] + "/" + horse_images[
                horse_path]
            images.append(str_path)
            train_labels.append(2)

# Stage 1.2 : Feature extraction
sift = cv2.xfeatures2d.SIFT_create()

dict_size = 15
BOW = cv2.BOWKMeansTrainer(dict_size)

for p in range(int(len(images))):
    random_image = images[p]
    img = cv2.imread(random_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    BOW.add(des)
    print('Processed image {} of {}'.format(p, len(images)))

# Stage 1.3 : Creation of visual vocabulary from Bag of Words
# visual vocabulary created
dictionary = BOW.cluster()

print("Dictionary Shape" , np.shape(dictionary))

# Flann based
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
sift2 = cv2.xfeatures2d.SIFT_create()
# using Brute force matcher
#bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(
# cv2.NORM_L2))

# using flann matcher
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, flann)
bowDiction.setVocabulary(dictionary)


for p in range(len(images)):
    im = cv2.imread(images[p])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    desc = bowDiction.compute(gray, sift.detect(gray))
    train_desc.extend(desc)

print("train sample size", len(train_labels))
print("train desc size", len(train_desc))

# Stage 1.4 : Training using KNearest Neighbour
# create the Knn object
knn = cv2.ml.KNearest_create()

# train with training_set
knn.train(np.array(train_desc).astype(np.float32), cv2.ml.ROW_SAMPLE,
          np.array(train_labels).astype(np.float32))


# Stage 2 : Testing the test images
for paths in range(len(path_test)):
    str_path = path_str_test + "/" + path_test[paths]
    images_test.append(str_path)

# plotting dimentions for pyplot
m,n = 5, 10

# testing the images for right label using trained KNN model
for path_p in range(len(images_test)):
    quo = int(path_p/5)
    rem = int(path_p%5)
    im = cv2.imread(images_test[path_p])
    imrgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    desc = bowDiction.compute(gray, sift2.detect(gray))
    ret, results, neighbours, dist = knn.findNearest(desc, 3)
    print("Image : ", images_test[path_p], " Label : ",
          label[int(results[0]) - 1])
    plt.subplot(m, n, quo * m + rem + 1), plt.imshow(imrgb, "gray")
    plt.title(label[int(results[0]) - 1]), plt.xticks([]), \
    plt.yticks([])

plt.show()
