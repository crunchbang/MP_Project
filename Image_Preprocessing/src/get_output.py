import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


cap = cv2.VideoCapture('dataset.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

while(1):
    ret, frame = cap.read()
    img = frame
    rgb = img[:,:,::-1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, otzu = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    h = otzu.shape[0]
    otzu[:int(h/5)] = 255
    kernel17 = np.ones((17, 17), np.uint8)
    closing17 = cv2.morphologyEx(otzu, cv2.MORPH_CLOSE, kernel17)
    opening17 = cv2.morphologyEx(closing17, cv2.MORPH_OPEN, kernel17)
    output = np.array(frame, copy=True)
    mask_inv = cv2.bitwise_not(opening17)
    road = cv2.bitwise_and(output, output, mask=mask_inv)

    # cv2.imshow("original", frame)
    # cv2.imshow("processed", road)

    out.write(frame_out)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

print("Done")
cap.release()
out.release()
cv2.destroyAllWindows()
