# ROADMAP
---------

* Try and isolate the road from the background noise.
* Extracting potholes from roads will be easier when the background has been eliminated.
* Test methods on relatively easy test images (where the demarcation between the road and pothole is clear).

## Methods

### Smoothing to reduce noise
* Gaussian Blur
* Median Blur

### Detect edges 
Detect edges or contours to isolate and extract the road boundary from the environment.
* Laplacian 
* Sobel
* Canny
* Contour Detection

### Clustering
Try to color quantize or segment the image to see if boundary between road and pothole can be identified.
* K-Means
* Mean Shift

