import os
import cv2



source = "rtsp://admin:q1234567@192.168.1.64/Streaming/Channels/1"
cam = cv2.VideoCapture(source)
img_counter = 0
while(cam.isOpened()):
    ret, frame = cam.read()
    cv2.imshow('frame', frame)
    if not ret:
        break
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
