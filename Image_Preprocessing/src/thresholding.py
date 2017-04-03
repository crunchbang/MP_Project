import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
 
def thresholding(img):
  #img = cv2.imread('/home/tarini/Documents/Pothole-Detection/Data/road_10.jpg',0)
  img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img1 = cv2.medianBlur(img1,5)
 
  ret,th1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
  th2 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
             cv2.THRESH_BINARY,11,2)
  th3 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
             cv2.THRESH_BINARY,11,2)
  blur = cv2.GaussianBlur(img1,(5,5),0)
  ret4,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
  titles = ['Original Image', 'Global Thresholding (v = 127)',
             'Otsus Thresholding' ,'Adaptive Gaussian Thresholding']
  images = [img, th1, th4, th3]
 
  for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
  plt.show()




mypath='/home/tarini/Documents/Pothole-Detection/Data'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
   images[n] = cv2.imread( join(mypath,onlyfiles[n]) )


for n in range(0,len(onlyfiles)):
  im = thresholding(images[n])

#global and otsu's thresholding have similar output, and adaptive thresholding seems to be it good job minus the noise, but for classic images.