"""Script to remove background from pictures

This script removes the background from pictures
and set it to any predefined color.

"""

from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

input_folder = "./img/input"
output_folder = "./img/output"

# Parameters
BLUR = 21
CANNY_THRESH_1 = 80
CANNY_THRESH_2 = 110
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (1.0,1.0,1.0) # In BGR format
# Constants
window_name = "Window Name"

def print_with_object(msg, obj):
    print(msg)
    print(obj)

def show_image_and_wait(img):
    cv2.imshow(window_name, img)
    cv2.waitKey()

def main():
    files = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
    print_with_object("list of files:", files)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for f in files:
        img = cv2.imread(join(input_folder, f))     # read image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert from one color space to another
        # Edge detection
        edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        show_image_and_wait(edges)
        # Find contours in edges, sort by area
        contour_info = []
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0]
        # Create empty mask, draw filled polygon on it corresponding to largest contour
        # Mask is black, polygon is white
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255))
        # Smooth mask, then blur it
        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
        mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
        show_image_and_wait(mask)
        # Blend masked img into MASK_COLOR background
        mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices,
        img         = img.astype('float32') / 255.0                 #  for easy blending
        masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
        masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit
        show_image_and_wait(masked)
        # Save
        cv2.imwrite(join(output_folder, f), masked)

if __name__ == "__main__":
    main()