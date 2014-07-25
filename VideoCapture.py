from threading import Thread
from cv2 import cv
import cv2
import numpy
import time
import Queue

green = (0, 255, 0) #green
red = (0, 0, 255) #red
blue = (254, 0, 0) #blue
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
    
    show_circles = False
    hough_circles_dp = 1
    hough_circles_minDist = 50
    hough_circles_param1 = 70
    hough_circles_param2 = 40

    show_lines = False
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

        if self.show_circles:
            circles = cv2.HoughCircles(edges, cv.CV_HOUGH_GRADIENT, self.hough_circles_dp, self.hough_circles_minDist, param1=self.hough_circles_param1, param2=self.hough_circles_param2)
        else:
            circles = None
        
        if circles != None and len(circles) > 0 and len(circles[0]) > 0:
            for circle in circles[0]:
                cv2.circle(image_small, (circle[0], circle[1]), circle[2], green, thickness=2, lineType=4, shift=0)
                cv2.circle(color_dst, (circle[0], circle[1]), circle[2], green, thickness=2, lineType=4, shift=0)
            cue_ball = self.find_cue_ball(image_small, circles[0])
            cv2.circle(image_small, (cue_ball[0], cue_ball[1]), cue_ball[2], red, thickness=2, lineType=4, shift=0)
            cv2.circle(color_dst, (cue_ball[0], cue_ball[1]), cue_ball[2], red, thickness=2, lineType=4, shift=0)

        if self.show_lines:
            lines = cv2.HoughLinesP(edges, self.hough_lines_rho, self.hough_lines_theta, threshold=self.hough_lines_threshold, minLineLength=self.hough_lines_minLineLength, maxLineGap=self.hough_lines_maxLineGap)
        else:
            lines = None
        
        if lines != None and len(lines) > 0 and len(lines[0]) > 0:
            for line in lines[0]:
                cv2.line(image_small, (line[0], line[1]), (line[2], line[3]), green, thickness=2, lineType=8, shift=0)
                cv2.line(color_dst, (line[0], line[1]), (line[2], line[3]), green, thickness=2, lineType=8, shift=0)
        return {"img1": image_small, "img2": color_dst}

    def find_cue_ball(self, image, circles):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightest_circle = None
        brightest_circle_brightness = 0
        for circle in circles:
            avg_brightness = self.calculate_avg_brightness(gray, (circle[0], circle[1]))
            if avg_brightness > brightest_circle_brightness:
                brightest_circle_brightness = avg_brightness
                brightest_circle = circle
        return circle
            
    def calculate_avg_brightness(self, image, circle):
        y_min = max(int(round(circle[0]))-10, 0)
        y_max = min(int(round(circle[0]))+10, len(image))
        x_min = max(int(round(circle[1]))-10, 0)
        x_max = min(int(round(circle[1]))+10, len(image[0]))
        count = 1
        gray_sum = 0
        for r in range(y_min, y_max):
            for c in range(x_min, x_max):
                gray_sum += image[r][c]
                count += 1
        return gray_sum / count

    def show_image(self, **kwargs):
        for key, value in kwargs.iteritems():
            cv2.imshow(key, value)
        cv2.waitKey(10)
    
    def set_lines(self, should_show):
        self.show_lines = should_show

    def set_circles(self, should_show):
        self.show_circles = should_show

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
