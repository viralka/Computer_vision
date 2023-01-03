import cv2
import numpy as np

# test

url = r"https://192.168.141.140:8080/video"
cap = cv2.VideoCapture(url)

while(True):
    camera, frame = cap.read()
    if frame is not None:
        cv2.imshow("Frame", frame)
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cv2.destroyAllWindows()