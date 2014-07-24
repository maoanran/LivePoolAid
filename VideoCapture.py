import cv2

vid = cv2.VideoCapture(0)
while True:
    retval, image =  vid.read()
    cv2.imshow('video capture', image)
