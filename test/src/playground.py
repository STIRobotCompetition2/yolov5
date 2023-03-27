import cv2
import torch
import numpy as np
from PIL import ImageGrab
# Image
im = ImageGrab.grab()  # take a screenshot

cam = cv2.VideoCapture(0)

model = torch.hub.load('/home/svenbecker/Documents/EPFL/Semester_Project/repositories/yolov5', 'custom', path="best-int8_edgetpu.tflite", source='local')  # local repo# Inference


while True:
    check, frame = cam.read()

    frame = frame[:320,:320,:]

    results = model(frame, size=320)



    for detection in results.xyxy[0]:
        if detection[4] < 0.1: continue
        corners = detection[0:4].tolist()
        corners = [int(val) for val in corners]
        cv2.rectangle(frame, pt1=corners[0:2], pt2=corners[2:4], color=[255,0,0,255],thickness=3)
    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()