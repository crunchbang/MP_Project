import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
    Images for each panorama is stored in the directory structure
    as show below:

    data
    ├── Panorama
       ├── One
       │   ├── institute11.jpg
       │   └── institute21.jpg
       └── Two
           ├── secondPic1.jpg
           └── secondPic2.jpg

"""
class Panorama:
    """ Panorama class for Generating Panorama images"""

    def __init__(self, location):
        self.location = location # image location
        self.images = []  # stores the images for creating panorama
        # descriptors for layout of pyplot
        self.count = 1
        self.m = 3
        self.n = 2

    # Load's images from a directory to a list and returns the list.
    def load_images(self):
        paths = os.listdir(self.location)
        for path in range(len(paths)):
            self.images.append(self.location + "/" + paths[
                path])

    def list_images(self):
        for img in range(len(self.images)):
            self.plot(self.m, self.n, self.images[img],
                      str("Image "+str(img + 1)))
            # self.show_plot()

    # converts bgr channel to gray channel
    def cvt_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # converts the bgr channel to rgb channel
    def cvt_bgr2rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # initializes the surf descriptor
    @staticmethod
    def surf(hessian):
        return cv2.xfeatures2d.SURF_create(hessianThreshold=hessian,
                                           upright=True,
                                           extended=True)

    # initialize flann matcher
    @staticmethod
    def flann():
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)

    # main logic goes here.
    def create_panorama(self):
        # initialize a surf detector with hessian as 400
        surf = self.surf(400)
        imgOne = cv2.imread(self.images[0])
        imgTwo = cv2.imread(self.images[1])

        grayOne = self.cvt_gray(imgOne)
        grayTwo = self.cvt_gray(imgTwo)

        # extract the keypoinst and descriptors for individaul images
        kpOne, desOne = surf.detectAndCompute(grayOne, None)
        kpTwo, desTwo = surf.detectAndCompute(grayTwo, None)

        imgOneU = cv2.drawKeypoints(imgOne, kpOne, None, (0, 127,
                                                          0),
                                    4)
        imgTwoU = cv2.drawKeypoints(imgTwo, kpTwo, None, (0, 127,
                                                          0),
                                    4)
        # initialize flann matcher
        flann = self.flann()
        matches = flann.knnMatch(np.array(desOne), np.array(desTwo),
                                 k=2)

        # store all the good matches
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        src_pts = np.float32([kpOne[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kpTwo[m.trainIdx].pt for m in good])

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                     5.0)

        im_out = cv2.warpPerspective(imgOne, M, (
            imgOne.shape[1] + imgTwo.shape[1],
            imgOne.shape[0]))
        im_out[0:imgTwo.shape[0], 0:imgTwo.shape[1]] = imgTwo

        self.plot_img(self.m, self.n, imgOneU, "Keypoints 1")
        self.plot_img(self.m, self.n, imgTwoU, "Keypoints 2")

        img3 = cv2.drawMatchesKnn(imgOne, kpOne, imgTwo, kpTwo,
                                  matches[:100], None,
                                  matchColor=(0, 127, 255), flags=2)
        self.plot_img(self.m, self.n, img3, "Matching Keypoints")
        self.plot_img(self.m, self.n, im_out, "Panorama")
        self.show_plot()

    def show_plot(self):
        # plt.show()
        # save the Panorama created in to the disk
        plt.savefig("Panorama" + str(self.count) + ".png",
                    bbox_inches="tight", dpi=200)

    # plot the image
    def plot_img(self, m, n, image, label):
        img = self.cvt_bgr2rgb(image)
        plt.subplot(m, n, self.count), plt.imshow(img), plt.xticks([
        ]), \
        plt.yticks([])
        plt.xlabel(label)
        self.count += 1

    # Reads an image from path and plots
    def plot(self, m, n, image, label):
        img = self.cvt_bgr2rgb(cv2.imread(image))
        plt.subplot(m, n, self.count), plt.imshow(img), plt.xticks([
        ]), \
        plt.yticks([])
        plt.xlabel(label)
        self.count += 1


def main():
    # creates an instance of the Panorama class
    instance = Panorama("data/Panorama/One")
    instance.load_images()
    instance.list_images()
    instance.create_panorama()


if __name__ == '__main__':
    main()
