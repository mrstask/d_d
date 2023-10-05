import csv
import os

import cv2

from detect_drone.filters import is_contour_large_enough
from detect_drone.helpers import key_function
from detect_drone.image_preprocessing import prepare_images

THRESHOLD_VALUE = 30
MAX_VALUE = 255
IMAGE_EXTENSION = ".jpg"
MIN_CONTOUR_AREA = 200


def detect_changes(background_gray, current_gray):
    diff = cv2.absdiff(current_gray, background_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def identify_and_show_changes(image_dir, background_filename, current_filename, min_contour_area=MIN_CONTOUR_AREA):
    background, current, background_gray, current_gray = prepare_images(image_dir, background_filename,
                                                                        current_filename)
    contours = detect_changes(background_gray, current_gray)
    tracker = None
    bbox = None
    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        if is_contour_large_enough(contour, min_contour_area):
            x, y, w, h = bounding_rect
            cv2.rectangle(current, (x, y), (x + w, y + h), (0, 255, 0), 2)
            tracker = cv2.TrackerKCF_create()
            tracker.init(current, (x, y, w, h))
            bbox = (x, y, w, h)
            break

    return tracker, bbox  # Return both tracker and bounding box


def load_frame(image_dir, filename):
    frame_path = os.path.join(image_dir, filename)
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, frame_gray


def detect_drone(background_gray, current_gray, min_contour_area=MIN_CONTOUR_AREA):
    contours = detect_changes(background_gray, current_gray)
    for contour in contours:
        if is_contour_large_enough(contour, min_contour_area):
            return cv2.boundingRect(contour)
    return None


def initialize_tracker(current, bbox):
    tracker = cv2.TrackerKCF_create()
    tracker.init(current, bbox)
    return tracker


def update_tracker(tracker, current):
    success, box = tracker.update(current)
    if success:
        bbox = tuple(int(v) for v in box)
        cv2.rectangle(current, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        return bbox, True
    return None, False


def redetect_drone(background_gray, current_gray, current):
    bbox = detect_drone(background_gray, current_gray)
    if bbox is not None:
        cv2.rectangle(current, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
        return bbox
    return None


def record_data(frame_idx, bbox, trajectory_data):
    trajectory_data.append([frame_idx, *bbox])


def write_data_to_csv(trajectory_data):
    with open('trajectory_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "X", "Y", "Width", "Height"])
        writer.writerows(trajectory_data)


def compare_frames_with_background(frames_dir: str, image_extension: str = ".jpg") -> None:
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(image_extension)], key=key_function)
    if not frames:
        raise FileNotFoundError(f"No images found in the directory: {frames_dir}")

    background_filename = frames.pop(0)
    background, background_gray = load_frame(frames_dir, background_filename)

    tracker = None
    trajectory_data = []

    for idx, frame in enumerate(frames):
        print(f"Processing frame: {frame}")
        current, current_gray = load_frame(frames_dir, frame)

        if tracker is None:
            print("Initializing tracker...")
            tracker, bbox = identify_and_show_changes(frames_dir, background_filename, frame)
        else:
            print("Updating tracker...")
            bbox, success = update_tracker(tracker, current)
            if not success:
                print("Tracker update failed. Re-detecting...")
                bbox = redetect_drone(background_gray, current_gray, current)
                if bbox is not None:
                    tracker = initialize_tracker(current, bbox)

        if bbox is not None:
            record_data(idx, bbox, trajectory_data)

        cv2.imshow("Current Image", current)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    write_data_to_csv(trajectory_data)


if __name__ == '__main__':
    compare_frames_with_background("frames")
