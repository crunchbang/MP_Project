import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

img = cv2.imread("road_9.jpg",0)
cv2.imshow("Image",img)

cv2.waitKey()
cv2.destroyAllWindows()