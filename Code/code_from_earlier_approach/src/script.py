import cv2
import numpy as np
from matplotlib import pyplot as plt

# Reading
img = cv2.imread("TrainData/Positive/G0011489.JPG",
                 cv2.IMREAD_COLOR)

# Resizing large image
resized = cv2.resize(img, None, fx=0.25, fy=0.25,
                     interpolation=cv2.INTER_LINEAR)
width, length, c = resized.shape
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# flatenning and drawing a rectangle
test = np.array(resized, copy=True)
cv2.rectangle(test, (300,300), (900,500), (0, 255, 0), 2)
roi = resized[300:500,300:900]

# HSV channel seperation
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# converting to a binary image
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
roi_gray = cv2.GaussianBlur(roi_gray,(5,5),0)
roi_gray = 255*(roi_gray.astype('uint8'))
ret,thresh = cv2.threshold(roi_gray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(roi, contours, -1, (0,255,0), 3)

# http://stackoverflow.com/questions/40741398/how-to-find-the-largest-contour
cnt = max(contours, key = cv2.contourArea)
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)

print(hull)

# http://stackoverflow.com/questions/41508775/drawing-convexhull-in-opencv2-python
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(roi,start,end,[0,255,0],2)
    cv2.circle(roi,far,5,[0,0,255],-1)

cv2.imshow("Image ROI", roi)
cv2.imshow("Image Contours", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(roi_gray)
# plt.show()