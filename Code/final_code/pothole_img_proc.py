import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from preprocess import extract_road
from preprocess import display_result


def blob_detection(im):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]),
                                          (255, 255, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imwrite("blod_detect.jpg", im_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contour_detection(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### GAUSSIAN FILTER
    gaussian = cv2.GaussianBlur(gray_img, (7, 7), 0, 0)

    sigma = 0.33
    v = np.median(gaussian)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    canny = cv2.Canny(gaussian, 100, 150)

    ### DILATION
    kernel = np.ones((15, 15), np.uint8)
    dilation = cv2.dilate(canny, kernel, iterations=4)

    ### CONTOUR DETECTION
    feature = img.copy()
    im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(feature, contours, -1, (255, 0, 255), 3)

    img_list = [gray_img, gaussian, canny, dilation, feature]
    labels = ["gray", "gaussian", "canny", "dilation", "feature"]
    display_result(img_list, labels)


if __name__ == "__main__":
    test = "Subset 1 (Simplex)/Test Data/"

    root = test
    images = os.listdir(root)
    for img_name in images:
        print("Image" + img_name)
        img = cv2.imread(root + img_name)
        output = extract_road(img_name, img)
        # contour_detection(output)
        # blob_detection(output)
