import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from preprocess import extract_road


def get_features_labels():
    path = 'potholes/'
    images = os.listdir(path)

    X = []
    y = []
    for img in images:
        print("Extracting from image " + img)
        pic = cv2.imread(path + img)
        pic = extract_road(img, pic)
        pic = cv2.resize(pic, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        X.append(pic.flatten())
        lbl = img.split('_')[0]
        y.append(lbl)
    print("Extraction done")
    print(len(X))
    return X, y


def train_classifier():
    print("Starting Training")
    X, y = get_features_labels()
    print("Extracted feature labels")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)

    # Trying out different classifiers
    y_pred_gaussian = classifier_gaussianNB(X_train, X_test, y_train, y_test)
    y_pred_knn = classifier_knn(X_train, X_test, y_train, y_test)
    y_pred_svm = classifier_svm(X_train, X_test, y_train, y_test)
    y_pred_decisionTree = classifier_decisionTree(X_train, X_test, y_train, y_test)
    y_pred_adaBoost = classifier_adaBoost(X_train, X_test, y_train, y_test)

    predictions = {'SVM': y_pred_svm, 'KNN': y_pred_knn,
                   'GaussianNB': y_pred_gaussian, 'Decision Tree': y_pred_decisionTree,
                   'AdaBoost using Decision Tree': y_pred_adaBoost}
    print("Prediction score")
    for key in predictions:
        print(key + ":" + str(accuracy_score(y_test, predictions[key]) * 100))


def classifier_knn(train_data, test_data, train_label, test_label, neigh=3):
    clf = KNeighborsClassifier(neigh)
    print("Training KNN")
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    return pred_label


def classifier_gaussianNB(train_data, test_data, train_label, test_label):
    clf = GaussianNB()
    print("Training Gaussian")
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    return pred_label


def classifier_svm(train_data, test_data, train_label, test_label):
    clf = SVC()
    print("Training SVM")
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    return pred_label


def classifier_decisionTree(train_data, test_data, train_label, test_label):
    clf = DecisionTreeClassifier()
    print("Training Decision Trees")
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    return pred_label


def classifier_adaBoost(train_data, test_data, train_label, test_label):
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
    print("Training AdaBoost")
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    return pred_label


if __name__ == "__main__":
    label_translate = {'N': 0, 'P': 1}
    train_classifier()
