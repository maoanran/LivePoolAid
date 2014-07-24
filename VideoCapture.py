import cv2
from cv2 import cv
import numpy
import time

HEIGHT = 500
WIDTH = HEIGHT * 16 / 9
color = (0, 0, 0) #black
vid = cv2.VideoCapture(0)

while True:
    retval, image =  vid.read()
    image_small = cv2.resize(image, (WIDTH, HEIGHT))
    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    print dir(gray)
    exit()
    blurred = cv2.GaussianBlur(gray, (9, 9), 2, 2) 
    circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, .1, param1=200, param2=100)
    if circles != None and len(circles) > 0 and len(circles[0]) > 0:
        for circle in circles[0]:
            cv2.circle(blurred, (circle[0], circle[1]), circle[2], color, thickness=2, lineType=4, shift=0)
    cv2.imshow('video capture', blurred)
    cv2.waitKey(10)

