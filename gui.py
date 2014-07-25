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
    global lines_checkbox
    global circles_checkbox
    lines_checkbox = IntVar()
    circles_checkbox = IntVar()

    for i in range(len(settings_vars)):
        settings.append(Setting(root, i, settings_vars[i]))

    Label(root, text='Lines').grid(row=len(settings_vars))
    Label(root, text='Circles').grid(row=len(settings_vars) + 1)
    Checkbutton(root, variable=lines_checkbox).grid(row=len(settings_vars), column=1)
    Checkbutton(root, variable=circles_checkbox).grid(row=len(settings_vars) + 1, column=1)

    Button(root, text='Update', command=update_callback).grid(row=len(settings_vars) + 2, column=1)

def update_callback():
    print 'Updating...'
    root.after(1, update_settings)

def execute_computer_vision():
    global tracker
    global root
    tracker.process_video()
    root.after(25, execute_computer_vision)

def update_settings():
    global tracker
    global lines_checkbox
    global circles_checkbox
    params = {}
    for setting in settings:
        if setting.entry.get() != '' and is_number(setting.entry.get()):
            params[setting.var_name] = setting.entry.get()
    tracker.update_settings(**params)

    tracker.set_lines(not not (lines_checkbox.get()))
    tracker.set_circles(not not (circles_checkbox.get()))

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
