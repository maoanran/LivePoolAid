import cv2
from cv2 import cv
import numpy

color = (0, 0, 255) #red
vid = cv2.VideoCapture(0)
while True:
    retval, image =  vid.read()
    image_small = cv2.resize(image, (533, 300))
    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2, 2) 
    circles = cv2.HoughCircles(blurred, cv.CV_HOUGH_GRADIENT, 1, .5, 200, 100)
    if circles != None:
        for circle in circles[0]:
            cv2.circle(blurred, (circle[0], circle[1]), circle[2], color)
    cv2.imshow('video capture', blurred)

