from itertools import combinations, product

class Validator:
    frame_list = []

    def __init__(self, function, num_frames=5, num_overlap=5):
       self.num_frames = num_frames
       self.num_overlap = num_overlap
       self.function = function

    def submit_frame(self, frame):
        self.frame_list.insert(0, frame)
        if len(self.frame_list) > self.num_frames:
            self.frame_list.pop()

    def validate(self):
        return self.function(self.frame_list, self.num_overlap)

    def set_num_frames(self, num_frames):
        self.num_frames = num_frames
        self.frame_list = self.frame_list[:num_frames]

    def set_num_overlap(self, num_overlap):
        self.num_overlap = num_overlap

    def set_num_overlap(self, num_overlap):
        self.num_overlap = num_overlap

def validate_circles(frame_list, num_overlap):
    result = []
    for combo in combinations(frame_list, num_overlap):
        intersect = combo[0]
        for i in range(len(combo)):
            intersect = compute_overlap(intersect,combo[i])
        result.extend(intersect)
    return filter_circles(result)


def filter_circles(lst):
    res = []
    for item in lst:
        if not in_list(item, res):
            res.append(item)
    return res

def in_list(item, lst):
    for other in lst:
        if item == other:
            return True
    return False

def compute_overlap(lst1, lst2):
    res = []
    for pair in product(lst1, lst2):
        if pair[0] == pair[1]:
            res.append(average(pair[0], pair[1]))
    return res

def average(c1, c2):
    return Circle([int((c1.x+c2.x)/2), int((c1.y+c2.y)/2), int((c1.radius+c2.radius)/2)])

def validate_lines(frame_list):
    pass

delta_x = 5
delta_y = 5
delta_radius = 3

class Circle(object):

    def __init__(self, arr):
        self.x = arr[0]
        self.y = arr[1]
        self.radius = arr[2]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            x_equal = self.x >= other.x - delta_x and self.x <= other.x + delta_x
            y_equal = self.y >= other.y - delta_y and self.y <= other.y + delta_y
            radius_equal = self.radius >= other.radius - delta_radius and self.radius <= other.radius + delta_radius
            return x_equal and y_equal and radius_equal
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
