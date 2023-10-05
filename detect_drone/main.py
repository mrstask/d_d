import os
from typing import Tuple

import cv2

from detect_drone.helpers import key_function

THRESHOLD_VALUE = 30
MAX_VALUE = 255
IMAGE_EXTENSION = ".jpg"
MIN_CONTOUR_AREA = 200


def is_contour_large_enough(contour, min_contour_area: int) -> bool:
    return cv2.contourArea(contour) >= min_contour_area


def prepare_images(image_dir, background_filename, current_filename):
    background_path = os.path.join(image_dir, background_filename)
    current_path = os.path.join(image_dir, current_filename)
    background = cv2.imread(background_path)
    current = cv2.imread(current_path)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return background, current, background_gray, current_gray


def detect_changes(background_gray, current_gray):
    diff = cv2.absdiff(current_gray, background_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def identify_and_show_changes(image_dir, background_filename, current_filename, min_contour_area=MIN_CONTOUR_AREA):
    background, current, background_gray, current_gray = prepare_images(image_dir, background_filename,
                                                                        current_filename)
    contours = detect_changes(background_gray, current_gray)
    tracker = None  # Initialize tracker to None
    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        if is_contour_large_enough(contour, min_contour_area):
            x, y, w, h = bounding_rect
            cv2.rectangle(current, (x, y), (x + w, y + h), (0, 255, 0), 2)
            tracker = cv2.TrackerKCF_create()  # Create a tracker instance, e.g., KCF tracker
            tracker.init(current, (x, y, w, h))  # Initialize tracker with initial bounding box

    cv2.imshow("Current Image", current)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return tracker


def load_frame(image_dir, filename):
    frame_path = os.path.join(image_dir, filename)
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, frame_gray


def compare_frames_with_background(frames_dir: str, image_extension: str = ".jpg") -> None:
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(image_extension)], key=key_function)
    if not frames:
        raise FileNotFoundError(f"No images found in the directory: {frames_dir}")

    background = frames.pop(0)
    tracker = None
    for frame in frames:
        current, current_gray = load_frame(frames_dir, frame)
        if tracker is None:
            tracker = identify_and_show_changes(frames_dir, background, frame)
        else:
            success, box = tracker.update(current)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(current, (x, y), (x + w, y + h), (0, 255, 0), 2)


if __name__ == '__main__':
    # extract_frames_from_video("4.mp4")
    compare_frames_with_background("frames")
