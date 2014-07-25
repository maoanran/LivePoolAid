from threading import Thread
from cv2 import cv
import cv2
import numpy
import time
import Queue
import math
import Validator
from Validator import Circle
import pdb

green = (0, 255, 0) #green
red = (0, 0, 255) #red
blue = (254, 0, 0) #blue
should_process = True

class CameraTracking:
    HEIGHT = 500
    WIDTH = HEIGHT * 16 / 9
    callback_queue = Queue.Queue()

    vid = cv2.VideoCapture(0)

    canny_threshold1 = 150
    canny_threshold2 = 200
    canny_apertureSize = 3
    canny_L2gradient = True

    show_circles = False
    hough_circles_dp = 1
    hough_circles_minDist = 50
    hough_circles_param1 = 40
    hough_circles_param2 = 20
    hough_circles_minRadius = 5
    hough_circles_maxRadius = 30

    show_lines = False
    hough_lines_rho = 1
    hough_lines_theta = cv.CV_PI / 180
    hough_lines_threshold = 100
    hough_lines_minLineLength = 10
    hough_lines_maxLineGap = 10

    cue_line_slope = 10
    cue_line_dist_max = 10
    cue_line_dist_min = 2

    circle_validator_frames = 2
    circle_validator_overlap = 2
    circle_validator_delta_x = 5
    circle_validator_delta_y = 5
    circle_validator_delta_radius = 3

    line_validator_frames = 2
    line_validator_overlap = 2

    rotate_angle = 0

    circle_validator = Validator.Validator(circle_validator_frames, circle_validator_overlap)
    line_validator = Validator.Validator(line_validator_frames, line_validator_overlap)

    def update_settings(self, **kwargs):
        for key, value in kwargs.iteritems():
            exec('self.' + key + ' = ' + value)

    def process_video(self):
        retval, image =  self.vid.read()

        image_small = cv2.resize(image, (self.WIDTH, self.HEIGHT))
        #projection = cv.CreateImage((self.WIDTH, self.HEIGHT), cv2.IPL_DEPTH_32F, 3)

        edges = cv2.Canny(image_small, self.canny_threshold1, self.canny_threshold2, apertureSize=self.canny_apertureSize, L2gradient=self.canny_L2gradient)
        color_dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if self.show_circles:
            circles = cv2.HoughCircles(edges, cv.CV_HOUGH_GRADIENT, self.hough_circles_dp, self.hough_circles_minDist,
                    param1=self.hough_circles_param1, param2=self.hough_circles_param2, minRadius=self.hough_circles_minRadius,
                    maxRadius=self.hough_circles_maxRadius)
        else:
            circles = None

        if circles != None and len(circles) > 0 and len(circles[0]) > 0:
            circle_class_list = list(Circle(circle) for circle in circles[0])
            Validator.delta_x = self.circle_validator_delta_x
            Validator.delta_y = self.circle_validator_delta_y
            Validator.delta_radius = self.circle_validator_delta_radius
            self.circle_validator.set_num_frames(self.circle_validator_frames)
            self.circle_validator.set_num_overlap(self.circle_validator_overlap)
            self.circle_validator.submit_frame(circle_class_list)
            circles_to_draw = self.circle_validator.validate()
            if len(circles_to_draw) > 0:
                for circle in circles_to_draw:
                    cv2.circle(image_small, (circle.x, circle.y), circle.radius, green, thickness=2, lineType=4, shift=0)
                    cv2.circle(color_dst, (circle.x, circle.y), circle.radius, green, thickness=2, lineType=4, shift=0)
                    # cv2.circle(projection, (circle.x, circle.y), circle.radius, blue, thickness=2, lineType=4, shift=0)
                cue_ball = self.find_cue_ball(image_small, circles_to_draw)
                cv2.circle(image_small, (cue_ball.x, cue_ball.y), cue_ball.radius, red, thickness=2, lineType=4, shift=0)
                cv2.circle(color_dst, (cue_ball.x, cue_ball.y), cue_ball.radius, red, thickness=2, lineType=4, shift=0)


        if self.show_lines:
            lines = cv2.HoughLinesP(edges, self.hough_lines_rho, self.hough_lines_theta,
                    threshold=self.hough_lines_threshold, minLineLength=self.hough_lines_minLineLength, maxLineGap=self.hough_lines_maxLineGap)
        else:
            lines = None

        if lines != None and len(lines) > 0 and len(lines[0]) > 0:
            for line in lines[0]:
                cv2.line(image_small, (line[0], line[1]), (line[2], line[3]), green, thickness=2, lineType=8, shift=0)
                cv2.line(color_dst, (line[0], line[1]), (line[2], line[3]), green, thickness=2, lineType=8, shift=0)

            cue_stick = self.find_cue_stick(lines[0], None)
            if cue_stick != None:
                centerline = self.create_center_line(cue_stick["line1"], cue_stick["line2"])
                path_lines = self.create_path_lines(centerline, lines[0])

                # Draw path
                for p_line in path_lines:
                    cv2.line(image_small, p_line[0], p_line[1], red, thickness=2, lineType=8, shift=0)
                    cv2.line(color_dst, p_line[0], p_line[1], red, thickness=2, lineType=8, shift=0)

                cv2.line(image_small, centerline[0], centerline[1], red, thickness=2, lineType=8, shift=0)
                cv2.line(color_dst, centerline[0], centerline[1], red, thickness=2, lineType=8, shift=0)
                # cv2.line(projection, centerline[0], centerline[1], green, thickness=2, lineType=8, shift=0)
                cv2.rectangle(image_small, cue_stick["point1"], cue_stick["point2"], blue, thickness=2, lineType=8, shift=0)
                cv2.rectangle(color_dst, cue_stick["point1"], cue_stick["point2"], blue, thickness=2, lineType=8, shift=0)

        #rotated = self.rotateImage(image_small, self.rotate_angle)

        return {"Original": image_small, "Edge Detection": color_dst}

    def create_path_lines(self, centerline, lines):
        path_lines = []
        centerline_slope = (centerline[0][1] - centerline[1][1]) * 1.0 / (centerline[0][0] - centerline[1][0])
        centerline_y_intercept = centerline[0][1] * 1.0 - centerline_slope * centerline[0][0]
        bounce_slope = -centerline_slope

        for line in lines:
            line_slope = (line[1] - line[3]) * 1.0 / (line[0] - line[2])
            line_angle = math.degrees(math.atan(line_slope))
            line_y_intercept = line[1] * 1.0 - line_slope * line[0]
            # If not vertical or horizontal
            if abs(line_angle - 90) > 5 or abs(line_angle) > 5:
                continue
            intersections_point = self.intersection(line_slope, line_y_intercept, centerline_slope, centerline_y_intercept)

            if(intersection_point[0] >= 0 and intersection_point[0] < (self.WIDTH - 1) and intersection_point[1] >= 0 and intersection_point[1] < (self.HEIGHT - 1)):
                # Find y intercept of bounced line
                bounce_y_intercept = intersection_point[1] * 1.0 - bounce_slope * intersection_point[0]
                path_lines.append( [ (0, int(bounce_y_intercept)), (int(self.WIDTH - 1), int(bounce_slope * (self.WIDTH - 1) + bounce_y_intercept)) ] )

        return path_lines

    def rotateImage(self, image, angle):
        image_center = tuple(numpy.array(image.shape) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
        return result


    def intersection(self, m1, b1, m2, b2):
        x = ((b2 - b1) / (m1- m2))
        y = m1 * x + b1
        return (x, y)


    def create_center_line(self, line, line2):
        point1 = (line[0], line[1])
        point2 = (line[2], line[3])
        point3 = (line2[0], line2[1])
        point4 = (line2[2], line2[3])

        line_angle = math.degrees(math.atan( float(line[1] - line[3]) / (line[0] - line[2]) ))
        line2_angle = math.degrees(math.atan( float(line2[1] - line2[3]) / (line2[0] - line2[2]) ))
        avg_angle = (line_angle + line2_angle) / 2
        if avg_angle > 89 and avg_angle < 91:
            avg_angle = 89.9

        print 'Average angle: ' + str(avg_angle)
        print 'Line1 angle: ' + str(line_angle)
        print 'Line2 angle: ' + str(line2_angle)

        # Get the perp line slope
        line1_perp_slope = (line[2] - line[0]) / (line[1] - line[3])
        line2_slope = (line[1] - line[3]) / (line[0] - line[2])

        # Find intersection point between line through some point on line1 with perp slope and line 2
        line1_perp_y_intercept = point1[1] - line1_perp_slope * point1[0]
        line2_y_intercept = point3[1] - line2_slope * point3[0]
        x = ((line2_y_intercept - line1_perp_y_intercept) / (line1_perp_slope - line2_slope))
        y = line1_perp_slope * x + line1_perp_y_intercept

        # This point is the point where the perp line intersects line2
        intersection_point = self.intersection(line1_perp_slope, line1_perp_y_intercept, line2_slope, line2_y_intercept)

        # Find midpoint of perp line
        mid_point = ((point1[0] + intersection_point[0]) / 2, (point1[1] + intersection_point[1]) / 2)

        # Calculate stuff for line function
        mid_point_slope = math.tan(math.radians(avg_angle))
        mid_point_y_intercept = mid_point[1] - mid_point_slope * mid_point[0]

        # Return line that stretches screen. y = m * x + b
        return [(0, int(mid_point_y_intercept)), (int(self.WIDTH - 1), int(mid_point_slope * (self.WIDTH - 1) + mid_point_y_intercept))]

    def find_cue_stick(self, lines, cue_ball):
        for line in lines:
            for line2 in lines:
                if not numpy.array_equal(line, line2):
                    line_angle = math.degrees(math.atan( (line[1] - line[3]) / (line[0] - line[2]) ))
                    line2_angle = math.degrees(math.atan( (line2[1] - line2[3]) / (line2[0] - line2[2]) ))
                    if abs(line_angle - line2_angle) < self.cue_line_slope:
                        dist1 = self.distance((line[0], line[1]), (line2[0], line2[1]))
                        dist2 = self.distance((line[0], line[1]), (line2[2], line2[3]))
                        dist3 = self.distance((line[2], line[3]), (line2[0], line2[1]))
                        dist4 = self.distance((line[2], line[3]), (line2[2], line2[3]))
                        if (dist1 < self.cue_line_dist_max and dist1 > self.cue_line_dist_min) \
                            or (dist2 < self.cue_line_dist_max and dist2 > self.cue_line_dist_min) \
                            or (dist3 < self.cue_line_dist_max and dist3 > self.cue_line_dist_min) \
                            or (dist4 < self.cue_line_dist_max and dist4 > self.cue_line_dist_min):
                            max_dist = max(dist1, dist2, dist3, dist4)
                            if max_dist == dist1:
                                return {"line1": line, "line2": line2, "point1": (line[0], line[1]), "point2": (line2[0], line2[1])}
                            elif max_dist == dist2:
                                return {"line1": line, "line2": line2, "point1": (line[0], line[1]), "point2": (line2[2], line2[3])}
                            elif max_dist == dist3:
                                return {"line1": line, "line2": line2, "point1": (line[2], line[3]), "point2": (line2[0], line2[1])}
                            else:
                                return {"line1": line, "line2": line2, "point1": (line[2], line[3]), "point2": (line2[2], line2[3])}

    def distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


    def find_cue_ball(self, image, circles):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightest_circle = None
        brightest_circle_brightness = 0
        for circle in circles:
            avg_brightness = self.calculate_avg_brightness(gray, circle)
            if avg_brightness >= brightest_circle_brightness:
                brightest_circle_brightness = avg_brightness
                brightest_circle = circle
        return brightest_circle

    def calculate_avg_brightness(self, image, circle):
        x_min = max(int(circle.x)-int(circle.radius), 0)
        x_max = min(int(circle.x)+int(circle.radius), len(image[0]) - 1)
        y_min = max(int(circle.y)-int(circle.radius), 0)
        y_max = min(int(circle.y)+int(circle.radius), len(image) - 1)
        count = 1
        bright_sum = 0
        for r in range(y_min, y_max):
            for c in range(x_min, x_max):
                bright_sum += image[r][c]
                count += 1
        return bright_sum # / count

    def show_image(self, **kwargs):
        for key, value in kwargs.iteritems():
            cv2.imshow(key, value)

    def set_lines(self, should_show):
        self.show_lines = should_show

    def set_circles(self, should_show):
        self.show_circles = should_show

if __name__ == "__main__":
    tracker = CameraTracking()
    while should_process:
        tracker.process_video()
