import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# downloaded folder contains DSC images that have been resized
# data folder contains the data obtained from google image search


def preprocess(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = img[:,:,::-1]
    #grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # blur_grey = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
    # blur_grey = cv2.medianBlur(hsv[:, :, 0], 11)
    ret, otzu = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # height, width, d = img.shape
    # mid_h = int(height / 2)
    # mid_w = int(width / 2)
    # size_h = 100
    # size_w = 300
    # roi = img[mid_h-size_h:, mid_w-size_w:mid_w+size_w]
    # h, w, c = roi.shape
    # cv2.rectangle(roi, (0, 0), (w, h), (0, 255, 0), 20)
    #img3 = cv2.Canny(grey, 100, 200)
    # blur_otzu = cv2.GaussianBlur(otzu, (3, 3), 0)
    #blur_otzu = cv2.medianBlur(otzu, 5)
    # img2, contours, hierarchy = cv2.findContours(otzu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # for cnts in contours:
    #     epsilon = 0.1 * cv2.arcLength(cnts, True)
    #     approx = cv2.approxPolyDP(cnts, epsilon, True)
    #     print(approx.shape)
    #     cv2.drawContours(img, [approx], -1, (255, 0, 0), 3)

    kernel17 = np.ones((17, 17), np.uint8)
    closing17 = cv2.morphologyEx(otzu, cv2.MORPH_CLOSE, kernel17)
    opening17 = cv2.morphologyEx(closing17, cv2.MORPH_OPEN, kernel17)
    output = np.array(rgb)
    #np.copy(output, rgb)
    output[opening17 == 255] = (255, 255, 255)
    canny = cv2.Sobel(output[:, :, 2], cv2.CV_32F, 0, 1, ksize=9)


    fig = plt.figure()
    # images = [rgb, hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2], grey, blur_grey, otzu, img, hsv]
    #images = [rgb, hsv[:, :, 2], otzu, opening17]
    images = [rgb, output, otzu, opening17, canny]
    #titles = ["orig", "value", "otzu", "dilation", "erosion", "opening", "closing", "final"]
    titles = ["orig", "output", "otzu",  "opening (erosion + dialation)", "canny"]
    row, col = (3, 2)

    for i in range(len(images)):
        ax = fig.add_subplot(row, col, i + 1)
        ax.set_title(titles[i])
        ax.imshow(images[i], cmap="gray")
        plt.axis("off")
    plt.show()



if __name__ == '__main__':
    #root_dir = '/Users/athul/src/MP_Project/Image_Preprocessing/Downloaded'
    #img_name = 'road_cam_32.jpg'
    # root_dir = 'downloaded'
    root_dir = 'data'
    images = os.listdir(root_dir)[1:]
    for img_name in images:
        print(img_name)
        path = os.path.join(root_dir, img_name)
        preprocess(path)
