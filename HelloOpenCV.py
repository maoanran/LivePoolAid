import cv2
import sys

if len(sys.argv) != 2:
    sys.exit("Expecting a single image file argument")
filename = sys.argv[1]

image = cv2.imread(filename)
print image.shape
image_small = cv2.resize(image, (800, 600))
textColor = (0, 0, 255)  # red
cv2.putText(image_small, "Hello World!!!", (200, 200),
            cv2.FONT_HERSHEY_PLAIN, 3.0, textColor,
            thickness=4, lineType=cv2.CV_AA)
cv2.imshow('Hello World GUI', image_small)
cv2.waitKey()
cv2.destroyAllWindows()
