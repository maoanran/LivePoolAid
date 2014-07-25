from Tkinter import *
from VideoCapture import CameraTracking

settings_vars = [
    'canny_threshold1',
    'canny_threshold2',
    'canny_apertureSize',
    'canny_L2gradient',
    'hough_circles_dp',
    'hough_circles_minDist',
    'hough_circles_param1',
    'hough_circles_param2',
    'hough_lines_rho',
    'hough_lines_theta',
    'hough_lines_threshold',
    'hough_lines_minLineLength',
    'hough_lines_maxLineGap',
]

settings = []
root = None
tracker = None

def main():
    global root
    global tracker
    tracker = CameraTracking()
    root = Tk()

    initWindow(root)
    initUI(root)
    root.after(25, execute_computer_vision)
    root.mainloop()

def initWindow(root):
    root.geometry('300x600+100+100')
    root.title('Live Pool Aid')

def initUI(root):
    for i in range(len(settings_vars)):
        settings.append(Setting(root, i, settings_vars[i]))

def execute_computer_vision():
    global tracker
    update_settings()
    tracker.process_video()
    root.after(25, execute_computer_vision)

def update_settings():
    global tracker
    params = {}
    for setting in settings:
        if setting.entry.get() != '' and is_number(setting.entry.get()):
            params[setting.var_name] = setting.entry.get()
    tracker.update_settings(**params)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class Setting:
    def __init__(self, root, num, var_name):
        self.var_name = var_name
        Label(root, text=var_name).grid(row=num)
        self.entry = Entry(root)
        self.entry.grid(row=num, column=1)

if __name__ == '__main__':
    main()
