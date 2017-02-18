import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


# scaling the image to same size and flattening the same, so that it acts as the feature
def img_resize(img, size=(32, 32)):
    return cv2.resize(img, size).flatten()


# getting the directory where training data is present
training_image_path = os.listdir("train")

training_labels = []
training_set = []
label_code = {'face': 0,
              'night': 1,
              'land': 2}

# Each file is labelled in the form
# className.number.jpeg
for image_name in training_image_path:
    # Extracts the training_labels
    label = image_name.split(".")[0]
    path = "train/" + image_name

    img = cv2.imread(path)
    pixels = img_resize(img)
    training_set.append(pixels)
    training_labels.append(label_code[label])

# convert everything into numpy array as needed by cv2.KNN
training_set = np.array(training_set).astype(np.float32)
training_labels = np.array(training_labels).astype(np.float32)

# read the test dataset
test_image_paths = os.listdir("test")
test_set = []

for image_name in test_image_paths:
    path = "test/" + image_name
    img = cv2.imread(path)
    pixels = img_resize(img)
    test_set.append(pixels)

code_label = {0: "Face",
              1: "Night",
              2: "Land"}

test_set = np.array(test_set).astype(np.float32)

# create the Knn object
knn = cv2.ml.KNearest_create()
# train with training_set
knn.train(training_set, cv2.ml.ROW_SAMPLE, training_labels)
# run on test_set with k = 5
ret, results, neighbours, dist = knn.findNearest(test_set, 3)

# display the result
fig = plt.figure()
pos = 1
for i in range(len(test_image_paths)):
    path = "test/" + test_image_paths[i]
    img = cv2.imread(path)
    # for displaying with matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    ax = fig.add_subplot(5, 5, pos)
    # code of the image from test_set
    result_code = results[i][0]
    ax.set_title(code_label[result_code])
    pos += 1
    ax.imshow(img, cmap="gray")
    plt.axis("off")

plt.show()
