import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def display_result(img_list, labels=None):
    fig = plt.figure()
    for i in range(len(img_list)):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.set_title(labels[i])
        ax.imshow(img_list[i], cmap="gray")
        plt.axis("off")
    plt.show()


def make_label_dict():
    test_labels = "Subset 1 (Simplex)/simpleTestFullSizeAllPotholesSortedFullAnnotation.txt"
    train_labels = "Subset 1 (Simplex)/simpleTrainFullPhotosSortedFullAnnotations.txt"

    paths = [test_labels, train_labels]
    label_dict = {}
    for p in paths:
        lines = [line.rstrip('\n') for line in open(p)]
        for line in lines:
            l = line.split(' ')
            label_dict[l[0]] = (l[1], l[2:])
    return label_dict


def box_pothole(image_name, image):
    # Draw white bounding box around potholes based on the txt file given.
    (n, coords) = coordinate_map[image_name]
    i = 0
    while i < 4 * int(n):
        top_left_x, top_left_y = int(coords[i]), int(coords[i + 1])
        w, h = int(coords[i + 2]), int(coords[i + 3])
        cv2.rectangle(image, (top_left_x, top_left_y), (top_left_x + w, top_left_y + h), (255, 255, 255), 15)
        i = i + 4


def get_roi(img, get_marked=False):
    width, length, c = img.shape
    roi_top_left_x = int(length / 2 + length / 12)
    roi_top_left_y = int(width - width / 8)
    roi_top_left = (roi_top_left_x, roi_top_left_y)

    roi_bottom_right_x = int(length / 2 + 2 * length / 12)
    roi_bottom_right_y = width - 50
    roi_bottom_right = (roi_bottom_right_x, roi_bottom_right_y)

    roi = img[roi_top_left_y:roi_bottom_right_y, roi_top_left_x:roi_bottom_right_x]

    marked_img = None
    if get_marked:
        # optional marking
        marked_img = np.array(img, copy=True)
        cv2.rectangle(marked_img, roi_top_left, roi_bottom_right, (0, 255, 0), 15)

    return roi, marked_img


def extract_road(img_name, img, verbose=False):
    # Based on trial and error
    # This cuts out the hood from all the pictures. WORKS!
    img = img[:1800]

    # Resize image; Can speed up computation but interferes with marking potholes
    # Coordinates in txt file are given w.r.t original resolution
    # resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # comment this and uncomment the above line when resized image is required
    resized = img

    original = np.array(img, copy=True)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    roi, img_roi_marked = get_roi(resized)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(roi_hsv)

    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    std_h = np.std(h)
    std_s = np.std(s)
    std_v = np.std(v)

    # Road color is assumed to be lying 3 std above and below each channel's mean (From paper)
    lower_threshold = np.array([mean_h - 3 * std_h, mean_s - 3 * std_s, mean_v - 3 * std_v])
    upper_threshold = np.array([mean_h + 3 * std_h, mean_s + 3 * std_s, mean_v + 3 * std_v])

    # Create mask based on calculated range
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    # Copy of the mask for finding contours (which modified the input image)
    mask_cpy = np.array(mask, copy=True)
    im2, contour, hierarchy = cv2.findContours(mask_cpy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Hull works with only one contour
    # Find the largest contour (Hopefully the road) and pass it to the cx.hull
    cnt = max(contour, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)  # , returnPoints=False)

    # Create mask from the hull coordinates
    new_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(new_mask, hull, 1)
    # Apply mask on original image to get road (Hopefully)
    result = cv2.bitwise_and(resized, resized, mask=new_mask)

    # To get an idea of where the actual potholes are
    # img_cpy = np.array(result, copy=True)
    # box_pothole(img_name, result)

    ## FOR GETTING OUTPUT
    # original2 = np.array(img, copy=True)
    # hull2 = cv2.convexHull(cnt, returnPoints=False)
    # defects = cv2.convexityDefects(cnt, hull2)
    # # http://stackoverflow.com/questions/41508775/drawing-convexhull-in-opencv2-python
    # for i in range(defects.shape[0]):
    #     s, e, f, d = defects[i, 0]
    #     start = tuple(cnt[s][0])
    #     end = tuple(cnt[e][0])
    #     far = tuple(cnt[f][0])
    #     cv2.line(original2, start, end, [0, 255, 0], 15)
    #     cv2.circle(original2, far, 15, [0, 0, 255], -1)
    #
    # original_rgb = np.array(img, copy=True)
    # original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
    # images = [original_rgb, original, img_roi_marked, roi, mask,  new_mask, original2, result]
    # titles = ["Original", "HSV", "Marked ROI", "ROI", "Mask", "Convex hull of largest contour", "Convex hull", "Result"]
    # for i in range(len(images)):
    #     plt.title(titles[i])
    #     plt.imshow(images[i], cmap="gray")
    #     plt.axis("off")
    #     plt.show()
    #     # plt.savefig("output/" + str(i) + "_" + titles[i] + '.png', bbox_inches="tight", pad_inches=0, dpi=300)

    if verbose:
        img_list = [original, roi, mask, new_mask, result]
        titles = ["Original", "ROI", "Mask", "Convex hull of largest contour", "Result"]
        display_result(img_list, titles)

    return result

if __name__ == "__main__":
    train_positive = "Subset 1 (Simplex)/Train Data/Positive Data/"
    train_negative = "Subset 1 (Simplex)/Train Data/Positive Data/"
    test = "Subset 1 (Simplex)/Test Data/"

    coordinate_map = make_label_dict()
    root = test
    images = os.listdir(root)
    for img_name in images:
        print("Image" + img_name)
        img = cv2.imread(root + img_name)
        extract_road(img_name, img)
