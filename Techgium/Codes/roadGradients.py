import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("smooth.jpg")
blur = cv2.GaussianBlur(img,(15, 15),0) 

# converting the image to a grayscale image
kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
sharp = cv2.filter2D(blur, -1, kernel)

gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# creating binary synthetic image
thresh, temp  = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# With change in ksize, thickness of edges change in x and y sobel derivatives
sobelX = cv2.Sobel(temp, cv2.CV_64F, 1, 0, ksize = 3)   # for x as 1 and y as 0
sobelY = cv2.Sobel(temp, cv2.CV_64F, 0, 1, ksize = 3)    # for x as 0 and y as 1
output = sobelX + sobelY

cv2.imshow("sobelX", sobelX)
cv2.imshow("sobelY", sobelY)
cv2.imshow("output", output)

#cv2.imwrite("sobelX.jpg", sobelX)
#cv2.imwrite("sobelY.jpg", sobelY)
cv2.imwrite("outputPotholes2.jpg", output)

# Plotting the output
plt.subplot(121),plt.imshow(sobelX), plt.title('sobelX')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(sobelY), plt.title('sobelY')
plt.xticks([]), plt.yticks([])

cv2.waitKey(0)
cv2.destroyAllWindows()