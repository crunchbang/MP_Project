import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth


def recreate_image(codebook, labels, h, w):
    """Replace image pixel with cluster center value"""

    d = codebook.shape[1]
    img = np.array(np.zeros((h, w, d)), dtype=np.float32)
    label_idx = 0
    for i in range(h):
        for j in range(w):
            img[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return img


def start_segmentation(image):
    """Segment image using KMeans"""

    original_bgr = cv2.imread(image)
    # Operations are done in HSV colorspace
    hsv_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    # Median blur to remove spurious noise and suppress minor details
    blur_img = cv2.medianBlur(hsv_img, 9)

    height, width, channel = blur_img.shape
    # Reshape image into Mx3 form for Mean Shift where M=height*width
    # Each row represents H,S,V values
    data = blur_img.reshape(height * width, channel)
    # Calculate bandwidth for Mean Shift using subset of data
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=100)
    ms.fit(data)
    # Recreate image from the MxN feature matrix
    # where each pixel is replaced by its cluster center
    result = recreate_image(ms.cluster_centers_, ms.labels_, height, width)

    # Laplacian edge detection to outline the segments
    lap_result = cv2.Laplacian(result, -1)
    # Superimpose the segments on original image
    for i in range(lap_result.shape[0]):
        for j in range(lap_result.shape[1]):
            if lap_result[i][j].any():
                lap_result[i][j] = [255, 255, 255]
                original[i][j] = [0, 255, 0]

    # For convenience of displaying
    lap_result = cv2.cvtColor(lap_result, cv2.COLOR_BGR2GRAY)

    # Display output
    fig = plt.figure()
    row = 1
    col = 2
    images = [original, lap_result]
    image_caption = ['Output', 'KMeans']

    for i in range(len(images)):
        ax = fig.add_subplot(row, col, i + 1)
        ax.imshow(images[i], cmap="gray")
        ax.set_title(image_caption[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()


if __name__ == '__main__':
    start_segmentation('c17.jpg')
