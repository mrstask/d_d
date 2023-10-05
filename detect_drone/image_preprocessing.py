import os

import cv2


def prepare_images(image_dir, background_filename, current_filename):
    background_path = os.path.join(image_dir, background_filename)
    current_path = os.path.join(image_dir, current_filename)
    background = cv2.imread(background_path)
    current = cv2.imread(current_path)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return background, current, background_gray, current_gray
