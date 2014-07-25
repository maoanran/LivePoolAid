from itertools import combinations, product

class Validator:
    frame_list = []

    def __init__(self, num_frames=5, num_overlap=5):
       self.num_frames = num_frames
       self.num_overlap = num_overlap

    def submit_frame(self, frame):
        self.frame_list.insert(0, frame)
        if len(self.frame_list) > self.num_frames:
            self.frame_list.pop()

    def validate(self):
        result = []
        for combo in combinations(self.frame_list, self.num_overlap):
            intersect = combo[0]
            for i in range(len(combo)):
                intersect = compute_overlap(intersect,combo[i])
            result.extend(intersect)
        return filter_list(result)

    def set_num_frames(self, num_frames):
        self.num_frames = num_frames
        self.frame_list = self.frame_list[:num_frames]

    def set_num_overlap(self, num_overlap):
        self.num_overlap = num_overlap

def filter_list(lst):
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
    if isinstance(c1, Circle):
        return Circle([int((c1.x+c2.x)/2), int((c1.y+c2.y)/2), int((c1.radius+c2.radius)/2)])
    else:
        return Line([int((c1.x1+c2.x1)/2), int((c1.y1+c2.y1)/2), int((c1.x2+c2.x2)/2), int((c1.y2+c2.y2)/2)])

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

class Line(object):

    def __init__(self, arr):
        self.x1 = arr[0]
        self.y1 = arr[1]
        self.x2 = arr[2]
        self.y2 = arr[3]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            x1_equal = self.x1 >= other.x1 - delta_x and self.x1 <= other.x1 + delta_x
            y1_equal = self.y1 >= other.y1 - delta_y and self.y1 <= other.y1 + delta_y
            x2_equal = self.x2 >= other.x2 - delta_x and self.x2 <= other.x2 + delta_x
            y2_equal = self.y2 >= other.y2 - delta_y and self.y2 <= other.y2 + delta_y
            return x1_equal and y1_equal and x2_equal and y2_equal
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)