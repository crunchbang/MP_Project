import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys, os
import cv2


# The data is organized as follows:
# data
# ├── test
# │   ├── bike_03.jpg
# │   ├── bike_69.jpg
# │   ├── bike_70.jpg
# │   ├── bike_75.jpg
# │   ├── horse_01.jpg
# │   ├── horse_02.jpg
# │   ├── ...
# │   ├── ...
# └── train
#     ├── bike
#     │   ├── bike_01.jpg
#     │   ├── bike_02.jpg
#     │   ├── bike_04.jpg
#     │   ├── bike_05.jpg
#     │   ├── bike_85.jpg
#     │   ├── ...
#     │   └── ...
#     └── horse
#         ├── horse_04.jpg
#         ├── horse_05.jpg
#         ├── horse_16.jpg
#         ├── horse_18.jpg
#         ├── ...
#         └── ...
# Train folder contains the training organized as folders with each folder representing a class.
# Test folder contains images from all classes.



class BagOfWords(object):
    """
    Implements Bag Of Words image classification
    Uses:
    * SURF : To detect keypoints and extract descriptors
    * KMeans : To cluster descriptors and form the vocabulary
    * Numpy.bincount : To create feature histogram
    * LogisticRegression : To classify the labeled feature historgrams
    """

    def __init__(self, n_clusters=15):
        """Initialize class with sane defaults"""
        self.train_label = []
        self.train_desc = np.array([])
        self.train_desc_size = []
        self.n_clusters = n_clusters
        self.features = np.array([])
        self.label_map = {}
        self.rev_label_map = {}
        self.surf = cv2.xfeatures2d.SURF_create(400)
        self.surf.setExtended(True)
        self.km = KMeans(n_clusters=self.n_clusters, random_state=0, n_jobs=-1)
        self.logistic = LogisticRegression(C=1e5)

    def extract_info(self, filepath):
        """Extract keypoints and descriptors using SURF"""
        file = os.path.basename(filepath)
        print('Info: Running extractor on image {}'.format(file))
        label = self.label_map[file.split('_')[0]]
        image = cv2.imread(filepath)
        kp, desc = self.surf.detectAndCompute(image, None)
        return label, desc, desc.shape[0]

    def surf_extract(self, train_path):
        """Create list of descriptors for all training images"""
        folders = os.listdir(train_path)
        for folderpath in folders:
            print('Info: Inside {}'.format(folderpath))
            files = os.listdir(os.path.join(train_path, folderpath))
            for f in files:
                print('Info: Process {}'.format(f))
                lbl, desc, dnum = self.extract_info(os.path.join(train_path, os.path.join(folderpath, f)))
                self.train_label.append(lbl)
                self.train_desc_size.append(dnum)
                if self.train_desc.size == 0:
                    self.train_desc = desc
                else:
                    self.train_desc = np.concatenate((self.train_desc, desc), axis=0)

    def create_vocabulary(self):
        """Create vocabulary by running K-Means on list of training descriptors"""
        print('Info: Running K-Means')
        self.km.fit(self.train_desc)
        labels = self.km.labels_
        print('Info: Generating Feature Histogram')
        num = len(self.train_desc_size)

        # create feature histogram for each image by separating descriptors list
        # into chunks for corresponding images
        chunk_end = 0
        for i in range(num):
            chunk_start = chunk_end
            chunk_end = chunk_end + self.train_desc_size[i]
            chunk = labels[chunk_start:chunk_end]
            feature_hist = np.bincount(chunk)
            if self.features.size == 0:
                self.features = feature_hist
            else:
                self.features = np.vstack((self.features, np.array(feature_hist)))

    def train(self, root):
        """Reads images, creates vocabulary and trains the classifier"""

        path_train = os.path.join(root, 'train')
        train_classes = os.listdir(path_train)
        print('Info: Starting')
        self.label_map = {lbl: idx for idx, lbl in enumerate(train_classes)}
        self.rev_label_map = {idx: lbl for idx, lbl in enumerate(train_classes)}
        print(self.label_map)
        self.surf_extract(path_train)
        print('Total training files : {}'.format(len(self.train_label)))
        print('Total training features : {}'.format(self.train_desc.shape[0]))
        print('Generating Vocabulary')
        self.create_vocabulary()
        print('Info: Vocabulary size {}'.format(self.features.shape[1]))
        print('Info: Training classifier')
        self.logistic.fit(self.features, self.train_label)

    def predict(self, path):
        """Extract descriptors from test image, create feature historgram and predicts the class"""
        print('Info: Starting prediction')
        print('Prediction - Actual')
        labels_true = []
        labels_pred = []
        files = os.listdir(path)
        for f in files:
            true_lbl = self.label_map[f.split('_')[0]]
            labels_true.append(true_lbl)
            image = cv2.imread(os.path.join(path, f))
            kp, desc = self.surf.detectAndCompute(image, None)
            labels = self.km.predict(desc)
            features_test = np.bincount(labels)
            pred_lbl = self.logistic.predict([features_test])
            print('{} - {}'.format(self.rev_label_map[pred_lbl[0]], self.rev_label_map[true_lbl]))
            labels_pred.append(pred_lbl[0])

        accuracy = accuracy_score(labels_true, labels_pred)
        print('Accuracy : {}'.format(accuracy))
        print('Total sample: {}'.format(len(labels_pred)))


if __name__ == "__main__":
    path_root = os.path.join(sys.path[0], 'data')
    bow = BagOfWords()
    print('Info: Data root - {}'.format(path_root))
    bow.train(path_root)
    bow.predict(os.path.join(path_root, 'test'))
