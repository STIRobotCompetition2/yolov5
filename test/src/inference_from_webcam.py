import cv2
import torch
import numpy as np
from PIL import ImageGrab
# Image
im = ImageGrab.grab()  # take a screenshot

cam = cv2.VideoCapture(0)

model = torch.hub.load('/home/svenbecker/Documents/EPFL/Semester_Project/yolov5', 'custom', path='model/best.pt', source='local')  # local repo# Inference

model.to("cuda")

while True:
    check, frame = cam.read()
    results = model(frame)
    results.print()
    for detection in results.xyxy[0]:
        if detection[4] < 0.6: continue
        corners = detection[0:4].tolist()
        corners = [int(val) for val in corners]
        cv2.rectangle(frame, pt1=corners[0:2], pt2=corners[2:4], color=[255,0,0,255],thickness=3)
    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()