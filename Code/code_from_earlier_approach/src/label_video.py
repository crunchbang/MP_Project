import cv2
import os

cap_data = cv2.VideoCapture('dataset.mp4')
cap_stable = cv2.VideoCapture('stable_output.mp4')
current_state = -1
ctr = 1
labels = { -1 : 'road', 1 : 'pothole'}

while (1):
    ret, frame_data = cap_data.read()
    ret, frame_stable = cap_stable.read()

    if current_state == 1:
        cv2.rectangle(frame_data, (0, 0), (20, 20), (0, 255, 0), 2)
    else:
        cv2.rectangle(frame_data, (0, 0), (20, 20), (0, 0, 255), 2)
    cv2.imshow("Frame", frame_data)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        current_state = current_state * -1

    f_name = str(ctr) + "_" + labels[current_state]
    cv2.imwrite(os.path.join('labelled_frames', f_name + '_data.jpg'), frame_data)
    cv2.imwrite(os.path.join('labelled_frames', f_name + '_stable.jpg'), frame_stable)

    ctr = ctr + 1

print("Done")
cap_data.release()
cv2.destroyAllWindows()
