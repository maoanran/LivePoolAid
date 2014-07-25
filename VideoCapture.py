from threading import Thread
from cv2 import cv
import cv2
import numpy
import time
import Queue

color = (0, 255, 0) #green
should_process = True

class CameraTracking:
    HEIGHT = 500
    WIDTH = HEIGHT * 16 / 9
    callback_queue = Queue.Queue()
   
    vid = cv2.VideoCapture(0)
    
    canny_threshold1 = 50
    canny_threshold2 = 150
    canny_apertureSize = 3
    canny_L2gradient = True
    
    show_circles = True
    hough_circles_dp = 1
    hough_circles_minDist = 50
    hough_circles_param1 = 70
    hough_circles_param2 = 40

    show_lines = True
    hough_lines_rho = 1
    hough_lines_theta = cv.CV_PI / 180
    hough_lines_threshold = 100
    hough_lines_minLineLength = 10
    hough_lines_maxLineGap = 10
    

    def update_settings(self, **kwargs):
        for key, value in kwargs.iteritems():
            exec('self.' + key + ' = ' + value)

    def process_video(self):
        retval, image =  self.vid.read()
        image_small = cv2.resize(image, (self.WIDTH, self.HEIGHT))

        edges = cv2.Canny(image_small, self.canny_threshold1, self.canny_threshold2, apertureSize=self.canny_apertureSize, L2gradient=self.canny_L2gradient)
        color_dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        #gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
        if self.show_circles:
            circles = cv2.HoughCircles(edges, cv.CV_HOUGH_GRADIENT, self.hough_circles_dp, self.hough_circles_minDist, param1=self.hough_circles_param1, param2=self.hough_circles_param2)
        else:
            circles = None

        if circles != None and len(circles) > 0 and len(circles[0]) > 0:
            for circle in circles[0]:
                cv2.circle(image_small, (circle[0], circle[1]), circle[2], color, thickness=2, lineType=4, shift=0)
                cv2.circle(color_dst, (circle[0], circle[1]), circle[2], color, thickness=2, lineType=4, shift=0)
        
        if self.show_lines:
            lines = cv2.HoughLinesP(edges, self.hough_lines_rho, self.hough_lines_theta, threshold=self.hough_lines_threshold, minLineLength=self.hough_lines_minLineLength, maxLineGap=self.hough_lines_maxLineGap)
        else:
            lines = None
        
        if lines != None and len(lines) > 0 and len(lines[0]) > 0:
            for line in lines[0]:
                cv2.line(image_small, (line[0], line[1]), (line[2], line[3]), color, thickness=2, lineType=8, shift=0)
                cv2.line(color_dst, (line[0], line[1]), (line[2], line[3]), color, thickness=2, lineType=8, shift=0)
        self.show_image(img1=image_small, img2=color_dst)
        #cv2.imshow('video capture', image_small)
        #cv2.imshow('edge detection', color_dst)
    
    def show_image(self, **kwargs):
        for key, value in kwargs.iteritems():
            cv2.imshow(key, value)
        cv2.waitKey(10)
    
    def toggle_lines(self):
        self.show_lines = !self.show_lines

    def toggle_circles(self):
        self.show_circles = !self.show_circles

    def main_loop():
        try:
            callback = callback_queue.get(false)
        except Queue.Empty:
            print 'queue empty!'
        callback()

if __name__ == "__main__":
    tracker = CameraTracking()
    while should_process:
        tracker.process_video()
